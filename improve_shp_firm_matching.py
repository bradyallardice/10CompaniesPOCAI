#!/usr/bin/env python3
"""
Improve SHP-to-X28 Firm Matching Coverage

Replaces the coauthor's R script (0_anonymous_firmid.R) matching logic with:
1. Enhanced lookup table from all available data sources
2. Core token matching (per coauthor's insight: word-level differences, not typos)
3. All ~200 manual fixes extracted from the R script

Two modes:
  --build-lookup: Assemble comprehensive lookup from all data sources (Phase 1)
  --match-names:  Match firm names against lookup (Phase 2, needs coauthor's file)
"""

import argparse
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
COAUTHOR_DATA_DIR = Path.home() / "Dropbox" / "kurer_allardice_technology" / "data"

# Legal suffixes to strip for normalization (Swiss business forms)
LEGAL_SUFFIXES = [
    "ag", "gmbh", "sa", "sàrl", "sarl", "ltd", "inc", "se",
    "genossenschaft", "stiftung", "verein", "anstalt",
    "gesellschaft", "holding", "group", "fondation",
    "association", "cooperative", "coopérative",
]

# Stopwords: common words in French/German/Italian that don't identify a company.
# These are stripped from core tokens to avoid false positives in token matching.
STOPWORDS = {
    # French articles, prepositions, conjunctions
    "de", "du", "des", "la", "le", "les", "un", "une",
    "et", "ou", "en", "au", "aux",
    "pour", "par", "sur", "avec", "dans", "sans", "sous",
    # German articles, prepositions, conjunctions
    "der", "die", "das", "den", "dem", "des",
    "und", "oder", "in", "im", "an", "am", "auf", "aus",
    "bei", "bis", "fuer", "fur", "mit", "nach", "von", "vom", "zu", "zum", "zur",
    "ueber", "unter",
    # Italian articles, prepositions
    "di", "il", "lo", "la", "gli", "le", "un", "una",
    "per", "con", "su", "tra", "fra", "da", "del", "dei", "della", "delle", "dello",
    # Generic descriptive words common in Swiss firm entries
    "service", "services", "centre", "center", "zentrum",
    "departement", "department", "departement", "abteilung",
    "direction", "direktion", "division",
    "bureau", "buero", "office",
    "section", "sektion",
    "institut", "institute",
    "ecole", "schule", "scuola", "school",
    "universite", "universitaet", "universita",
    "hopital", "spital", "ospedale", "hospital",
    "canton", "kanton", "cantone",
    "commune", "gemeinde", "comune",
    "ville", "stadt", "citta",
    "region", "regional", "regionale",
    "national", "nationale", "nationaler",
    "federal", "federale",
    "suisse", "schweiz", "svizzera", "swiss",
    "international", "internationale", "internationaler",
    "public", "publique", "oeffentlich",
    "prive", "privat", "privata",
    "general", "generale", "generaler",
}

# Placeholder firm names to filter out
PLACEHOLDER_NAMES = {"", "-", "--", "---", "----", ".", "....", "na", "n/a"}


def anonymize_firm_id(company_id: int) -> int:
    """Convert company_id to anonymized firm_id: (company_id + 13) * 13"""
    return (company_id + 13) * 13


def deanonymize_firm_id(firm_id: int) -> int:
    """Recover company_id from anonymized firm_id: (firm_id / 13) - 13"""
    if firm_id % 13 != 0:
        raise ValueError(f"firm_id {firm_id} is not divisible by 13")
    return (firm_id // 13) - 13


# ============================================================
# Manual Fixes (extracted from 0_anonymous_firmid.R)
# ============================================================

def get_manual_fixes() -> pd.DataFrame:
    """
    All ~200 manual fixes from the coauthor's R script (0_anonymous_firmid.R).
    Each entry: (firm_name, firm_loc, company_id).
    firm_loc can be None for name-only matches.
    """
    fixes = []

    def add(name, loc, company_id):
        fixes.append({"firm_name": name, "firm_loc": loc, "company_id": company_id})

    # --- ROUND 1: Major employers with department/building names ---

    # HUG - Hôpitaux Universitaires de Genève
    add("HUG Hôpitaux Universitaires de Genève", "Genève", 40003401)

    # SRO Spital Langenthal
    add("SRO Spital Langenthal", "Langenthal", 40000914)

    # Inselspital
    add("Inselspital", "Bern", 40001165)
    add("Inselspital Bern", "Bern", 40001165)

    # Luzerner Kantonsspital LUKS (company_id=40003429, firm_id=520044746)
    add("Luzerner Kantonsspital LUKS Luzern", "Luzern", 40003429)
    add("Luzerner Kantonsspital LUKS Luzern Gärtnerei", "Luzern", 40003429)
    add("Luzerner Kantonsspital LUKS Sursee", "Sursee", 40003429)
    add("Luzerner Kantonsspital LUKS Wolhusen", "Wolhusen", 40003429)

    # PricewaterhouseCoopers
    add("PricewaterhouseCoopers AG", "Zürich", 40002024)

    # Bruker BioSpin AG
    add("Bruker BioSpin AG", "Fällanden", 40000186)

    # ETH Zürich
    add("ETH Zürich   Campus Science City", "Zürich", 40003782)
    add("ETH Zürich   ETH Zentrum", "Zürich", 40003782)
    add("ETH Zürich   UniversitätsSpital Zürich", "Zürich", 40003782)
    add("ETH Zürich    ETH Zentrum", "Zürich", 40003782)

    # Novartis Pharma
    add("Novartis Pharma AG Werk St. Johann/Campus", "Basel", 40123051)

    # Swisscom (Schweiz) AG Fixnet -> maps to Swisscom subsidiary initially
    # but later consolidated to main Swisscom (40001940) in post-processing
    add("Swisscom (Schweiz) AG Fixnet FWS Managementgebäude", "Olten", 40001940)

    # Banque Cantonale Vaudoise
    add("Banque Cantonale Vaudoise CAB Centre administratif", "Prilly", 40000138)

    # Credit Suisse
    add("Credit Suisse AG Credit Suisse Financial Services", "Zürich", 40001573)
    add("Credit Suisse AG Private Banking", "Zürich", 40001573)
    add("Credit Suisse AG Private Banking Zürich-Uetlihof A/B + ZN", "Zürich", 40001573)
    add("Credit Suisse (Schweiz) AG", "Zürich", 40001573)

    # Nestlé
    add("Nestlé S.A.", "Vevey", 40003804)

    # Zürcher Kantonalbank
    add("Zürcher Kantonalbank Geschäftshaus Hard A", "Zürich", 40003787)

    # --- ROUND 2: Additional hospitals and public broadcasters ---

    add("Hôpital du Jura Site de Delémont", "Delémont", 40015124)
    add("Fondation Les Perce-Neige", "Les Hauts-Geneveys", 40016195)
    add("Gesundheitsdepartement Universitätsspital Basel inkl. Frauenklinik", "Basel", 40000516)
    add("Spital Bülach", "Bülach", 40011565)
    add("RTS Radio Télévision Suisse", "Genève", 40014995)
    add("Televisione svizzera di lingua lingua italiana", "Comano", 40109073)
    add("Mobilière Suisse Société d'assurances sur la vie SA", "Nyon", 40003298)
    add("Les hôpitaux universitaires de Genève (HUG DEAS)", "Genève", 40003401)
    add("Credit Suisse AG Corporate & Retail Banking Zürich-Uetlihof A/B + ZN", "Zürich", 40001573)

    # --- ROUND 3: Transport ---
    add("Verkehrsbetriebe Zürich Büroneubau Hauptgebäude", "Zürich", 40003827)

    # --- ROUND 5: UBS variants (all → 40001137) ---
    ubs_id = 40001137

    # UBS Switzerland AG with various locations
    for loc in ["Zürich", "Basel", "Lausanne", "Lugano", "Birmensdorf ZH",
                 "8001 Zürich", "Aarau", "Morges", "Neuchâtel", "Pfäffikon SZ",
                 "Zollikon", "Davos Platz", "Gland", "Glattbrugg", "Klosters",
                 "Renens VD", "Winterthur", "Zug"]:
        add("UBS Switzerland AG", loc, ubs_id)
    add("UBS Switzerland AG Filiale Albisrieden Dorf", "Zürich", ubs_id)

    # UBS AG with building names
    for name in ["UBS AG WDR, Haus 2", "UBS AG WDR, Haus 1"]:
        add(name, "Glattbrugg", ubs_id)
    for name in ["UBS AG Dinocenter", "UBS AG VF Trakt A / Flur Süd",
                  "UBS AG Felsenhof", "UBS AG Flurhof FB",
                  "UBS AG Hochhaus zur Schanzenbrücke", "UBS AG Flur-Nord Trakt A",
                  "UBS AG Hauptsitz", "UBS AG kappeli", "UBS AG Bahnhofstrassetrakt",
                  "UBS AG Flurhof FC", "UBS AG Flur-Nord Trakt B", "UBS AG Flurhof FD",
                  "UBS AG Löwenplatz", "UBS AG Vza1"]:
        add(name, "Zürich", ubs_id)
    add("UBS AG Migros", "Sursee", ubs_id)
    add("UBS AG  Geschäftsleitung", "Basel", ubs_id)
    add("UBS AG  Geschäftsstelle", "Arbon", ubs_id)

    # UBS SA (French)
    for loc in ["Renens VD", "Lausanne", "Manno", "Lugano", "Neuchâtel",
                 "Carouge GE", "La Chaux-de-Fonds", "Chiasso", "Morges", "Nyon"]:
        add("UBS SA", loc, ubs_id)
    add("UBS SA Centre des Acacias", "Carouge GE", ubs_id)
    add("UBS SA City Nuova", "Lugano", ubs_id)
    add("UBS SA Palazzo Mercurio", "Chiasso", ubs_id)

    # UBS short names
    add("UBS", "Zürich", ubs_id)
    add("UBS", "Zürich altstätten", ubs_id)

    # UBS subsidiaries
    add("UBS LIMITED, London, Swiss Branch, Opfikon", "Glattbrugg", ubs_id)
    add("UBS Asset Management AG", "Zürich", ubs_id)
    add("UBS Kulturstiftung", "Zürich", ubs_id)

    # --- ROUND 6: Post CH / Die Post variants ---
    post_id = 40003062  # Post CH AG (firm_id = 520039975)

    for name in [
        "Die Post Poststellen und Verkauf", "Die Post Funktionsbereiche",
        "Die Post Briefpost", "Die Post Paketpost",
        "Die Post Paketpost Region Mitte", "Die Post PostMail",
        "Die Post Konzerndienste", "Die Post PostMail Stammhaus",
        "Die Post Briefpost Ostermundigen Z", "Die Post",
        "Die Post Information Technology Services", "Die Post AG",
        "Post CH AG PostMail", "Post CH AG Management",
        "Post CH AG PostNetz", "Post CH AG Postlogistics",
        "Post CH AG Poststellen und Verkauf",
        "Post CH AG Réseau postal et vente",
        "SV (Schweiz) AG Personalrestaurant Information Technologie Service - Die Post",
    ]:
        add(name, None, post_id)

    # PostFinance (separate company)
    postfinance_id = 40016183
    for name in [
        "Die Post PostFinance", "Die Post PostFinance St.Gallen AD GK",
        "Die Post Postfinance VZ Netstal", "Die Post Postfinance Bern PF PK/KD",
        "Die Post Postfinance Münchenst PF OC",
    ]:
        add(name, None, postfinance_id)

    # --- ROUND 7: Migros Aare and Ostschweiz ---
    migros_aare_id = 40115178
    for name in [
        "Genossenschaft Migros Aare Verwaltung",
        "Genossenschaft Migros Aare M Oensingen Lebensmittel",
        "Genossenschaft Migros Aare MM Unterentfelden Lebensmittel",
        "Genossenschaft Migros Aare MMM Brügg-Centre Brügg",
        "Genossenschaft Migros Aare SportXX Buchs",
        "Genossenschaft Migros Aare MM Biel City Center Lebensmittel",
        "Genossenschaft Migros Aare M Muri be Lebensmittel",
        "Genossenschaft Migros Aare Zentralverwaltung",
        "Genossenschaft Migros Aare M Solothurn Märet Hauptbahnhof",
        "Genossenschaft Migros Aare MM Solothurn Gurzelngasse",
        "Genossenschaft Migros Aare Sportxx Burgdorf",
        "Genossenschaft Migros Aare MM Schönbühl Shoppyland",
        "Genossenschaft Migros Aare Lebensmittel",
        "Genossenschaft Migros Aare MM Aarau-Telli Lebensmittel",
        "Genossenschaft Migros Aare MMM Langenthal Lebensmittel",
    ]:
        add(name, None, migros_aare_id)

    migros_ost_id = 40115180
    for name in [
        "Genossenschaft Migros Ostschweiz MMM St. Gallen-Neumarkt",
        "Genossenschaft Migros Ostschweiz D+G Buchs SG Mparc",
        "Genossenschaft Migros Ostschweiz MM Winterthur - Rosenberg",
        "Genossenschaft Migros Ostschweiz Hausbäckerei Gossau",
        "Genossenschaft Migros Ostschweiz MM Uzwil - Steinacker",
        "Genossenschaft Migros Ostschweiz OBI Winterthur - Grüzepark",
        "Genossenschaft Migros Ostschweiz Logistik",
        "Genossenschaft Migros Ostschweiz M Münchwilen",
        "Genossenschaft Migros Ostschweiz MMM Mels - Pizolpark",
        "Genossenschaft Migros Ostschweiz D+G Rheinpark",
        "Genossenschaft Migros Ostschweiz GG Altenrhein Stadler",
        "Genossenschaft Migros Ostschweiz M Hinwil",
        "Genossenschaft Migros Ostschweiz M Multiplex",
        "Genossenschaft Migros Ostschweiz MElectronics Wetzikon",
        "Genossenschaft Migros Ostschweiz MFIT Buchs - Activ Fitness",
        "Genossenschaft Migros Ostschweiz MM Widnau - Rhydorf-Center",
        "Genossenschaft Migros Ostschweiz Chickeria St. Gallen Bohl",
        "Genossenschaft Migros Ostschweiz Gastronomie GG FHS",
        "Genossenschaft Migros Ostschweiz M Sirnach",
        "Genossenschaft Migros Ostschweiz MElectronics - Grüzepark",
        "Genossenschaft Migros Ostschweiz MElectronics Amriswil -Amriville",
        "Genossenschaft Migros Ostschweiz Micasa Mels - Pizolpark",
    ]:
        add(name, None, migros_ost_id)

    # --- ROUND 8: Credit Suisse remaining variants ---
    cs_id = 40001573

    # With location
    cs_loc_fixes = [
        ("Credit Suisse AG Private Banking Zürich-Uetlihof C1", "Zürich"),
        ("Credit Suisse AG Credit Suisse Banking", "Zürich"),
        ("Credit Suisse Funds AG", "Zürich"),
        ("Credit Suisse Fleetmanagement AG", "Glattpark (Opfikon)"),
        ("Credit Suisse Group AG", "Zürich"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Dübendorf"),
        ("Credit Suisse AG Credit Suisse Financial Services Zürich-Uetlihof A/B + ZN", "Zürich"),
        ("Pensionskasse der Credit Suisse Group (Schweiz)", "Zürich"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Genève"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Glattbrugg"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Lausanne"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Locarno"),
        ("Credit Suisse AG Private & Corporate Clients", "Zürich"),
        ("Credit Suisse Group", "Zürich"),
        ("Credit Suisse (Schweiz) Hypotheken AG", "Zürich"),
        ("Credit Suisse AG Information Technology Alla Bolla", "Giubiasco"),
        ("Credit Suisse AG Private Banking", "Bern"),
        ("Credit Suisse AG Private Banking Zürich-Uetlihof D", "Zürich"),
        ("Credit Suisse Asset Management Funds", "Zürich"),
        ("Credit Suisse (Schweiz) AG", "Kloten"),
        ("Credit Suisse AG Private Banking", "Lugano"),
        ("Credit Suisse AG Private Banking", "Rapperswil SG"),
        ("DSR Personalrestaurant Credit Suisse Tower", "Zürich"),
        ("Credit Suisse", "Zürich"),
        ("Credit Suisse (Schweiz) AG", "Basel"),
        ("Credit Suisse (Schweiz) AG", "Brig"),
        ("Credit Suisse (Schweiz) AG", "Bülach"),
        ("Credit Suisse (Schweiz) AG", "Davos Platz"),
        ("Credit Suisse (Schweiz) Hypotheken AG   Credit Suisse (Schweiz) AG", "Zürich"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Basel"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Glattpark (Opfikon)"),
        ("Credit Suisse AG Credit Suisse Financial Services Zürich-Uetlihof C1", "Zürich"),
        ("Credit Suisse AG Private Banking", "Solothurn"),
        ("Credit Suisse AG Private Banking", "St. Moritz"),
        ("Credit Suisse Credit Suisse Financial Service", "Genève"),
        ("Credit Suisse Credit Suisse Financial Service", "Zürich"),
        ("Credit Suisse Private Banking Zürich-Uetlihof D", "Zürich"),
        ("Credit Suisse", "Zürich-oerlikon"),
        ("Credit Suisse (Schweiz) AG", "Aarau"),
        ("Credit Suisse (Schweiz) AG", "Brugg AG"),
        ("Credit Suisse (Schweiz) AG", "Genève"),
        ("Credit Suisse (Schweiz) AG", "Rheinfelden"),
        ("Credit Suisse (Schweiz) AG", "Weinfelden"),
        ("Credit Suisse AG Credit Suisse Financial Services", "Chiasso"),
        ("Credit Suisse AG Credit Suisse Financial Services Zürich-Uetlihof C2", "Zürich"),
        ("Credit Suisse AG Credit Suisse Financial Services Zürich-Uetlihof RZ 4", "Zürich"),
        ("Credit Suisse AG Information Technology", "Zürich"),
        ("Credit Suisse AG Pension Funds CH", "Sion"),
        ("Credit Suisse AG Private Banking", "St. Gallen"),
        ("Credit Suisse AG Zürich-Uetlihof EZ", "Zürich"),
        ("Credit Suisse AG, Credit Suisse Financial Services", "Zürich"),
        ("Credit Suisse Schweiz AG", "9500 Wil SG"),
        ("Credit Suisse Zürich-Uetlihof A/B + Zn", "Zürich"),
    ]
    for name, loc in cs_loc_fixes:
        add(name, loc, cs_id)

    # --- ROUND 9: FHNW ---
    fhnw_id = 40003345
    for name in [
        "Fachhochschule Nordwestschweiz FHNW Pädagogische Hochschule",
        "Fachhochschule Nordwestschweiz FHNW Hochschule für Technik",
        "Fachhochschule Nordwestschweiz FHNW Campus Muttenz",
        "Fachhochschule Nordwestschweiz FHNW Hochschule für Wirtschaft",
        "Fachhochschule Nordwestschweiz FHNW Direktion und Services",
        "Fachhochschule Nordwestschweiz FHNW Campus Olten",
        "Fachhochschule Nordwestschweiz FHNW Hochschule für Soziale Arbeit",
        "Fachhochschule Nordwestschweiz FHNW PH Weiterbildung und Beratung",
        "FHNW Pädagogische Hochschule",
        "Fachhochschule Nordwestschweiz",
        "Fachhochschule Nordwestschweiz FHNW HGK Bildene Kunst Medienkunst",
        "Fachhochschule Nordwestschweiz FHNW PH ISP",
    ]:
        add(name, None, fhnw_id)

    # --- ROUND 10: More Migros (Aare, Zürich, Luzern) ---
    for name in [
        "Genossenschaft Migros Aare ME Baden",
        "Genossenschaft Migros Aare MM Aarau-Igelweid Lebensmittel",
        "Genossenschaft Migros Aare MM Huttwil Lebensmittel",
        "Genossenschaft Migros Aare MR Zofingen Mrest Zofingen",
        "Genossenschaft Migros Aare Betriebszentrale",
        "Genossenschaft Migros Aare M Langenthal",
        "Genossenschaft Migros Aare M Murten",
        "Genossenschaft Migros Aare M Solothurn",
        "Genossenschaft Migros Aare MM Bern",
        "Genossenschaft Migros Aare MM Biel",
        "Genossenschaft Migros Aare SportXX",
        "Genossenschaft Migros Aare Micasa",
        "Genossenschaft Migros Aare Do it",
    ]:
        add(name, None, migros_aare_id)

    migros_zh_id = 40114475
    for name in [
        "Genossenschaft Migros Zürich Direktion",
        "Genossenschaft Migros Zürich M Zürich-Airport",
        "Genossenschaft Migros Zürich MM Volketswil",
        "Genossenschaft Migros Zürich MM Zürich-Affoltern",
        "Genossenschaft Migros Zürich M Näfels",
        "Genossenschaft Migros Zürich MM Zürich-Altstetten",
        "Genossenschaft Migros Zürich M Bülach",
        "Genossenschaft Migros Zürich MM Zürich-Letzipark",
        "Genossenschaft Migros Zürich M Zug",
        "Genossenschaft Migros Zürich MM Dietikon",
        "Genossenschaft Migros Zürich SportXX",
        "Genossenschaft Migros Zürich Micasa",
        "Genossenschaft Migros Zürich Do it",
        "Genossenschaft Migros Zürich Logistik",
    ]:
        add(name, None, migros_zh_id)

    migros_lu_id = 40113351
    for name in [
        "Genossenschaft Migros Luzern M Ruopige Zentrum Lebensmittel & Non Food",
        "Genossenschaft Migros Luzern M Horw Lebensmittel & Non Food",
        "Genossenschaft Migros Luzern DIY & Garden Sursee-Park",
        "Genossenschaft Migros Luzern MICASA",
        "Genossenschaft Migros Luzern MM Hochdorf-Seetal-Center Lebensmittel & Non Food",
        "Genossenschaft Migros Luzern MM Luzern-Neustadt Lebensmittel & Non Food",
        "Genossenschaft Migros Luzern M Kriens Lebensmittel & Non Food",
        "Genossenschaft Migros Luzern M Emmen Lebensmittel & Non Food",
        "Genossenschaft Migros Luzern SportXX",
        "Genossenschaft Migros Luzern Do it",
        "Genossenschaft Migros Luzern MM Sursee",
        "Genossenschaft Migros Luzern Direktion",
    ]:
        add(name, None, migros_lu_id)

    # --- ROUND 11: SBB, Nespresso, Coop ---
    sbb_id = 40003529
    for name in [
        "Ferrovie Federali Svizzere FFS",
        "Schweizerische Bundesbahnen SBB operation center 1",
        "Schweizerische Bundesbahnen SBB  Verwaltung",
        "Chemins de fer fédéraux suisses CFF Administration",
        "Schweizerische Bundesbahnen SBB  Betrieb",
        "SBB Cargo Elsässertor", "SBB",
        "Chemins de Fer Fédéraux Suisses CFF Administration",
        "Chemins de fer fédéraux suisse CFF",
        "Schweizerische Bundesbahnen SBB",
        "Chemins de fer fédéraux suisses CFF",
        "SBB Cargo",
        "Schweizerische Bundesbahnen SBB Personaldienst",
    ]:
        add(name, None, sbb_id)

    nespresso_id = 40058790
    for name in [
        "Nestlé Nespresso SA", "Nestlé Nespresso SA Romont",
        "Nestlé Nespresso SA Avenches", "Nestlé Nespresso SA Orbe",
    ]:
        add(name, None, nespresso_id)

    coop_id = 40001171
    for name in [
        "Coop VZ & ZB Wangen Logistik, Administration",
        "Coop CSC Volkiland", "Coop CL Industrie", "Coop Betriebszentrale",
        "Coop CL Wattwil", "Coop Bau + Hobby", "Coop CC Obermatt",
        "Coop Coop City Gerbergasse", "Coop Ipermercato Cattori",
        "Coop Logistique, Informatik Aclens", "Coop Genossenschaft",
        "Coop Supermarkt", "Coop City", "Coop Pronto",
        "Coop Megastore", "Coop Restaurant",
    ]:
        add(name, None, coop_id)

    # --- ROUND 12: Major Swiss corporations ---
    roche_id = 40001150
    for name in [
        "Roche Holding AG", "F. Hoffmann-La Roche AG Hauptbetrieb",
        "F. Hoffmann-La Roche AG", "Roche Pharma AG", "Roche Diagnostics AG",
    ]:
        add(name, None, roche_id)

    swisscom_id = 40001940
    for name in [
        "Swisscom Banking Provider AG",
        "Swisscom (Suisse) SA Fixnet FWS",
        "Swisscom (Schweiz) AG Swisscom Shop",
        "Swisscom (Svizzera) SA Fixnet FWS",
        "Swisscom IT Services AG", "Swisscom Directories AG",
        "Swisscom (Schweiz) AG Fixnet",
        "Swisscom (Suisse) SA", "Swisscom (Svizzera) SA",
    ]:
        add(name, None, swisscom_id)

    swiss_re_id = 40128521
    for name in [
        "Swiss Reinsurance Company Ltd",
        "Swiss Re Foundation    c/o Swiss Re AG",
        "Schweizerische Rückversicherungs-Gesellschaft Swiss Re Next",
        "Swiss Re Life Capital Management Ltd, Swiss Re Next",
        "Pensionskasse Schweizerische Rückversicherungs-Gesellschaft (Swiss Re)",
        "Swiss Re AG", "Swiss Re Management AG",
    ]:
        add(name, None, swiss_re_id)

    abb_id = 40001712
    for name in [
        "ABB Power Grids Switzerland AG",
        "ABB Schweiz AG CMC Low Voltage Products",
        "ABB Schweiz AG Werk Baden", "ABB Schweiz AG Werk Baden - Dättwil",
        "ABB Schweiz AG", "ABB Sécheron SA", "ABB Turbo Systems AG",
        "ABB Switzerland Ltd",
    ]:
        add(name, None, abb_id)

    richemont_id = 40115374
    for name in [
        "VACHERON CONSTANTIN, Branch of Richemont International SA Manufacture",
        "IWC SCHAFFHAUSEN, Branch of Richemont International SA",
        "PIAGET, Branch of Richemont International SA Horlogerie",
        "CARTIER HORLOGERIES Branch of Richemont International SA",
        "OFFICINE PANERAI, Branch of Richemont International SA",
        "A. Lange & Söhne, Branch of Richemont International SA",
        "BAUME & MERCIER, Branch of Richemont International SA",
    ]:
        add(name, None, richemont_id)

    givaudan_id = 40174086
    for name in [
        "Givaudan Suisse SA", "Givaudan Schweiz AG",
        "Givaudan Schweiz AG  Aromen & Geschmackstoffe",
    ]:
        add(name, None, givaudan_id)

    # --- ROUND 13: RUAG, La Poste, AMAG, hospitals ---
    ruag_id = 40121164
    for name in [
        "RUAG Schweiz AG", "RUAG AG", "RUAG Defence",
        "RUAG Ammotec", "RUAG Space", "RUAG Aviation",
    ]:
        add(name, None, ruag_id)

    add("La Poste Réseau postal et vente", None, post_id)

    amag_id = 40001163
    for name in [
        "AMAG Automobil- und Motoren AG", "AMAG Group AG", "AMAG Import AG",
    ]:
        add(name, None, amag_id)

    add("Kantonsspital Münsterlingen", None, 40002053)
    add("Spital Thurgau AG  Kantonsspital Münsterlingen", None, 40002053)
    add("Spital Limmattal Personalabteilung", None, 40000905)
    add("Spital Limmattal", None, 40000905)
    add("Spital Grabs Spitalregion Rheintal Werdenberg Sarganserland", None, 40011109)
    add("Spitalregion Rheintal Werdenberg Sarganserland", None, 40011109)

    # --- ROUND 14: Common abbreviations and short forms ---
    # SBB / CFF / FFS short forms (main company 40003529)
    sbb_id = 40003529
    for name in [
        "SBB", "SBB AG", "SBB Cargo", "SBB Immobilien", "SBB Transportpolizei",
        "SBB AG Informatik", "SBB Restaurant WylerPark",
        "CFF", "CFF SA", "FFS",
    ]:
        add(name, None, sbb_id)

    # HUG short forms (40003401)
    hug_id = 40003401
    for name in [
        "HUG", "HUG Secrétariat général", "HUG Hôpital des Enfants",
        "HUG Département de l'enfant et de l'adolescent",
        "HUG Belle-Idée", "HUG Département de médecine",
    ]:
        add(name, None, hug_id)

    # NOTE: EPFL is not in X28 data (federal institution, no job vacancies)
    # Cannot be matched to a company_id

    # Swisscom subsidiaries → main Swisscom (40001940)
    swisscom_id = 40001940
    for name in [
        "Swisscom Grosskundenunternehmung", "Swisscom Event & Media Solutions",
        "Swisscom ITS Workplace Services", "Swisscom Broadcast AG",
        "Swisscom Event & Media Solutions AG",
    ]:
        add(name, None, swisscom_id)

    # La Poste / Schweizerische Post variants → Post CH AG (40003062)
    for name in [
        "Schweizerische Post", "La Poste Courrier", "La Poste Suisse",
        "La Posta", "La Poste",
    ]:
        add(name, None, post_id)

    # --- ROUND 15: Raiffeisen local branches + Vögele Shoes ---
    # All Raiffeisen local branches/cooperatives → Raiffeisen Gruppe (40001284)
    raiffeisen_id = 40001284
    for name, loc in [
        ("Banque Raiffeisen des Montagnes Neuchâteloises société coopérative", "Le Locle"),
        ("Banque Raiffeisen d'Yverdon-les-Bains société coopérative", "Yverdon-les-Bains"),
        ("Banca Raiffeisen del Monte San Giorgio società cooperativa", "Ligornetto"),
        ("Banque Raiffeisen de Massongex/St-Maurice/Vérossaz société coopérative", "Massongex"),
        ("Banque Raiffeisen Mont-Aubert Orbe société coopérative", "Montagny-près-Yverdon"),
        ("Banque Raiffeisen du Gros-de-Vaud société coopérative", "Echallens"),
        ("Banque Raiffeisen Martigny et Région, société coopérative", "Martigny"),
        ("Banque Raiffeisen du Val-de-Travers société coopérative", "Fleurier"),
        ("Banque Raiffeisen Lausanne-Haute-Broye-Jorat société coopérative", "Lausanne"),
        ("Banque Raiffeisen Lausanne-Haute-Broye- Jorat société coopérative", "Lausanne"),
        ("Banque Raiffeisen du Mont-Tendre société coopérative", "Montricher"),
        ("Banque Raiffeisen Entremont société coopérative", "Le Châble VS"),
        ("Banque Raiffeisen d'Assens société coopérative", "Cheseaux-sur-Lausanne"),
        ("Banca Raiffeisen del Camoghè società cooperativa", "Giubiasco"),
        ("Banque Raiffeisen Plateau du Jorat-Molondin société coopérative", "Thierrens"),
        ("Banque Raiffeisen Genève Ouest-Meyrin, société coopérative", "Genève"),
    ]:
        add(name, loc, raiffeisen_id)

    # Vögele Shoes in Coop-Park → Karl Vögele AG (40001041)
    add("Vögele Shoes Coop-Park", "Affoltern am Albis", 40001041)

    # --- ROUND 16: Coop/Migros store-level matches (from web LLM classification) ---
    coop_migros_file = Path(__file__).parent / "Data" / "shp_matching" / "coop_migros_matched.csv"
    if coop_migros_file.exists():
        df_cm = pd.read_csv(coop_migros_file)
        df_cm = df_cm[df_cm["matched_entity"].notna()]
        n_cm = 0
        for _, row in df_cm.iterrows():
            add(row["P__W52_NAME"], row["P__W52_LOC"] if pd.notna(row["P__W52_LOC"]) else None, int(row["matched_entity"]))
            n_cm += 1
        logger.info(f"  Round 16: Loaded {n_cm} Coop/Migros fixes from {coop_migros_file.name}")

    df = pd.DataFrame(fixes)
    df["firm_id"] = df["company_id"].apply(anonymize_firm_id)
    logger.info(f"Extracted {len(df)} manual fixes covering {df['company_id'].nunique()} unique companies")
    return df


# ============================================================
# Name Normalization
# ============================================================

def normalize_name(name: str) -> str:
    """Normalize a firm name for matching."""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    # Normalize unicode
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e")
    s = s.replace("à", "a").replace("â", "a")
    s = s.replace("ô", "o").replace("î", "i").replace("ù", "u")
    # Remove punctuation (keep spaces and alphanumeric)
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_core_tokens(name: str) -> list:
    """Extract core tokens from a normalized name, removing legal suffixes and stopwords."""
    if not name:
        return []
    tokens = name.split()
    core = [t for t in tokens
            if t not in LEGAL_SUFFIXES
            and t not in STOPWORDS
            and len(t) > 1]
    return core


# ============================================================
# Phase 1: Build Enhanced Lookup
# ============================================================

def build_lookup(args) -> None:
    """Build enhanced lookup table from all data sources."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract manual fixes ---
    logger.info("=" * 60)
    logger.info("Step 1: Extracting manual fixes from R script...")
    manual_fixes = get_manual_fixes()
    manual_fixes_path = output_dir / "manual_fixes.csv"
    manual_fixes.to_csv(manual_fixes_path, index=False)
    logger.info(f"  Saved {len(manual_fixes)} manual fixes to {manual_fixes_path}")

    # --- Step 2: Load data sources ---
    logger.info("=" * 60)
    logger.info("Step 2: Loading data sources...")

    # Source A: scrape_uid_complete.csv (all name→UID variants)
    scrape_path = Path(args.scrape_file)
    if not scrape_path.exists():
        raise FileNotFoundError(f"Scrape file not found: {scrape_path}")
    scrape = pd.read_csv(scrape_path)
    logger.info(f"  Source A (scrape): {len(scrape)} rows, "
                f"{scrape['bur_uid'].notna().sum()} with UIDs")

    # Source B: shp_x28_match.csv (UID→company_id + X28 names)
    x28_match_path = Path(args.x28_match_file)
    if not x28_match_path.exists():
        raise FileNotFoundError(f"X28 match file not found: {x28_match_path}")
    x28_match = pd.read_csv(x28_match_path)
    # Rename 'id' to 'company_id' for clarity
    if "id" in x28_match.columns:
        x28_match = x28_match.rename(columns={"id": "company_id"})
    logger.info(f"  Source B (x28_match): {len(x28_match)} rows")

    # Source C: shp_uid_final.csv
    uid_final_path = Path(args.uid_final_file)
    if not uid_final_path.exists():
        raise FileNotFoundError(f"UID final file not found: {uid_final_path}")
    uid_final = pd.read_csv(uid_final_path)
    logger.info(f"  Source C (uid_final): {len(uid_final)} rows")

    # Source D: company_mapping.csv
    company_mapping_path = Path(args.company_mapping)
    if not company_mapping_path.exists():
        raise FileNotFoundError(f"Company mapping not found: {company_mapping_path}")
    company_mapping = pd.read_csv(company_mapping_path)
    logger.info(f"  Source D (company_mapping): {len(company_mapping)} rows")

    # --- Step 3: Build UID→company_id mapping from X28 ---
    logger.info("=" * 60)
    logger.info("Step 3: Building UID→company_id mapping...")

    uid_to_company = {}
    for _, row in x28_match.iterrows():
        uid = row.get("uid")
        cid = row.get("company_id")
        if pd.notna(uid) and pd.notna(cid):
            uid_str = str(uid).strip()
            if uid_str:
                uid_to_company[uid_str] = int(cid)
    logger.info(f"  {len(uid_to_company)} unique UID→company_id mappings")

    # --- Step 4: Assemble all name variants ---
    logger.info("=" * 60)
    logger.info("Step 4: Assembling name variants from all sources...")

    lookup_entries = []

    def add_entry(name_original, location, company_id, company_name, uid, source):
        if pd.isna(name_original) or not str(name_original).strip():
            return
        name_orig = str(name_original).strip()
        name_norm = normalize_name(name_orig)
        if not name_norm:
            return
        loc = str(location).strip() if pd.notna(location) and str(location).strip() else None
        cname = str(company_name).strip() if pd.notna(company_name) else ""
        uid_str = str(uid).strip() if pd.notna(uid) else ""
        lookup_entries.append({
            "name_normalized": name_norm,
            "name_original": name_orig,
            "location": loc,
            "company_id": int(company_id),
            "company_name": cname,
            "uid": uid_str,
            "source": source,
        })

    # Source A: All scrape variants linked to X28 via UID
    n_scrape_linked = 0
    for _, row in scrape.iterrows():
        uid = row.get("bur_uid")
        if pd.isna(uid):
            continue
        uid_str = str(uid).strip()
        company_id = uid_to_company.get(uid_str)
        if company_id is None:
            continue
        company_name = company_mapping.loc[
            company_mapping["company_id"] == company_id, "company_name"
        ].values
        cname = company_name[0] if len(company_name) > 0 else ""

        # Add both mis_firmname and bur_firmname as variants
        add_entry(row.get("mis_firmname"), row.get("mis_firmloc"),
                  company_id, cname, uid_str, "scrape_mis")
        add_entry(row.get("bur_firmname"), row.get("bur_firmloc"),
                  company_id, cname, uid_str, "scrape_bur")
        n_scrape_linked += 1
    logger.info(f"  Source A: {n_scrape_linked} scrape rows linked to X28")

    # Source B: X28 company names directly
    for _, row in x28_match.iterrows():
        cid = row.get("company_id")
        if pd.isna(cid):
            continue
        add_entry(row.get("name"), row.get("address"),
                  int(cid), row.get("name"), row.get("uid"), "x28_name")

    # Source C: uid_final variants
    for _, row in uid_final.iterrows():
        uid = row.get("uid")
        if pd.isna(uid):
            continue
        uid_str = str(uid).strip()
        company_id = uid_to_company.get(uid_str)
        if company_id is None:
            continue
        company_name = company_mapping.loc[
            company_mapping["company_id"] == company_id, "company_name"
        ].values
        cname = company_name[0] if len(company_name) > 0 else ""
        add_entry(row.get("mis_firmname"), row.get("mis_firmloc"),
                  company_id, cname, uid_str, "uid_final_mis")
        add_entry(row.get("bur_firmname"), row.get("bur_firmloc"),
                  company_id, cname, uid_str, "uid_final_bur")

    # Source D: company_mapping (canonical X28 names)
    for _, row in company_mapping.iterrows():
        cid = row.get("company_id")
        cname = row.get("company_name")
        if pd.notna(cid) and pd.notna(cname):
            add_entry(cname, None, int(cid), cname, "", "company_mapping")

    # Source E: Manual fixes
    for _, row in manual_fixes.iterrows():
        add_entry(row["firm_name"], row["firm_loc"],
                  row["company_id"], "", "", "manual_fix")

    logger.info(f"  Total raw entries: {len(lookup_entries)}")

    # --- Step 5: Deduplicate and flag ambiguities ---
    logger.info("=" * 60)
    logger.info("Step 5: Deduplicating and flagging ambiguities...")

    df_lookup = pd.DataFrame(lookup_entries)

    # Drop exact duplicates on (name_normalized, location, company_id)
    df_lookup = df_lookup.drop_duplicates(
        subset=["name_normalized", "location", "company_id"]
    )

    # Flag ambiguous names (same normalized name → different company_ids)
    name_to_companies = df_lookup.groupby("name_normalized")["company_id"].nunique()
    ambiguous_names = set(name_to_companies[name_to_companies > 1].index)
    df_lookup["is_ambiguous"] = df_lookup["name_normalized"].isin(ambiguous_names)

    n_ambiguous = len(ambiguous_names)
    n_total = df_lookup["name_normalized"].nunique()
    logger.info(f"  Unique normalized names: {n_total}")
    logger.info(f"  Ambiguous (>1 company_id): {n_ambiguous}")
    logger.info(f"  Unambiguous: {n_total - n_ambiguous}")
    logger.info(f"  Total lookup entries: {len(df_lookup)}")
    logger.info(f"  Unique companies in lookup: {df_lookup['company_id'].nunique()}")

    # --- Step 6: Save ---
    lookup_path = output_dir / "enhanced_lookup.csv"
    df_lookup.to_csv(lookup_path, index=False)
    logger.info(f"  Saved lookup to {lookup_path}")

    # Summary by source
    logger.info("\n  Entries by source:")
    for source, count in df_lookup["source"].value_counts().items():
        logger.info(f"    {source}: {count}")

    # Sample ambiguous names
    if n_ambiguous > 0:
        logger.info(f"\n  Sample ambiguous names (max 10):")
        for name in list(ambiguous_names)[:10]:
            companies = df_lookup.loc[
                df_lookup["name_normalized"] == name, "company_id"
            ].unique()
            logger.info(f"    '{name}' → {list(companies)}")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 complete. Enhanced lookup table built.")
    logger.info("=" * 60)


# ============================================================
# Phase 2: Match Names
# ============================================================

def match_names(args) -> None:
    """Match firm names against enhanced lookup."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load lookup
    lookup_path = Path(args.lookup_file)
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file not found: {lookup_path}. Run --build-lookup first.")
    df_lookup = pd.read_csv(lookup_path)
    logger.info(f"Loaded lookup: {len(df_lookup)} entries, "
                f"{df_lookup['company_id'].nunique()} companies")

    # Load manual fixes
    manual_fixes_path = Path(args.manual_fixes_file)
    if not manual_fixes_path.exists():
        raise FileNotFoundError(f"Manual fixes file not found: {manual_fixes_path}")
    manual_fixes = pd.read_csv(manual_fixes_path)
    logger.info(f"Loaded {len(manual_fixes)} manual fixes")

    # Load input names
    names_path = Path(args.names_file)
    if not names_path.exists():
        raise FileNotFoundError(f"Names file not found: {names_path}")
    df_names = pd.read_csv(names_path)
    logger.info(f"Loaded {len(df_names)} input names")

    # Detect columns
    name_col = None
    loc_col = None
    for c in df_names.columns:
        cl = c.lower().strip()
        if cl in ("firm_name", "p__w52_name", "firmname", "name"):
            name_col = c
        elif cl in ("firm_location", "firm_loc", "p__w52_loc", "firmloc", "location"):
            loc_col = c

    if name_col is None:
        raise ValueError(
            f"Cannot find name column. Columns: {list(df_names.columns)}. "
            f"Expected one of: firm_name, P__W52_NAME, firmname, name"
        )
    logger.info(f"  Name column: '{name_col}', Location column: '{loc_col}'")

    # Preprocess input
    df_names = df_names.copy()
    df_names["firm_name_clean"] = df_names[name_col].astype(str).str.strip()
    df_names["firm_loc_clean"] = (
        df_names[loc_col].astype(str).str.strip() if loc_col else None
    )
    # Filter placeholders
    mask = df_names["firm_name_clean"].str.lower().isin(PLACEHOLDER_NAMES)
    n_placeholder = mask.sum()
    if n_placeholder > 0:
        logger.info(f"  Filtered {n_placeholder} placeholder names")
        df_names = df_names[~mask].copy()

    logger.info(f"  {len(df_names)} names to match")

    # Normalize input
    df_names["name_normalized"] = df_names["firm_name_clean"].apply(normalize_name)
    df_names["name_tokens"] = df_names["name_normalized"].apply(get_core_tokens)

    # Build lookup indices
    logger.info("Building lookup indices...")

    # Manual fixes index: (name, loc) → company_id and name → company_id
    mf_nameloc = {}
    mf_nameonly = defaultdict(set)
    for _, row in manual_fixes.iterrows():
        name = str(row["firm_name"]).strip()
        loc = str(row["firm_loc"]).strip() if pd.notna(row["firm_loc"]) else None
        cid = int(row["company_id"])
        if loc:
            mf_nameloc[(name.lower(), loc.lower())] = cid
        mf_nameonly[name.lower()].add(cid)

    # Normalized lookup index: name_normalized → [(company_id, company_name, location)]
    norm_index = defaultdict(list)
    for _, row in df_lookup.iterrows():
        norm_index[row["name_normalized"]].append({
            "company_id": int(row["company_id"]),
            "company_name": row.get("company_name", ""),
            "location": row.get("location"),
        })

    # Token index: token → set of (company_id, name_normalized)
    token_index = defaultdict(set)
    for _, row in df_lookup.iterrows():
        tokens = get_core_tokens(row["name_normalized"])
        cid = int(row["company_id"])
        for t in tokens:
            token_index[t].add((cid, row["name_normalized"]))

    # Get company_name mapping
    cid_to_name = {}
    for _, row in df_lookup.iterrows():
        cid = int(row["company_id"])
        cname = row.get("company_name", "")
        if cid not in cid_to_name and pd.notna(cname) and str(cname).strip():
            cid_to_name[cid] = str(cname).strip()

    # --- Matching ---
    results = []
    unmatched_indices = set(df_names.index)

    # Methods that always need manual review (fuzzy/token-based)
    REVIEW_METHODS = {"token_exact", "token_exact_loc", "token_relaxed", "fuzzy_char"}

    def record_match(idx, company_id, method, score):
        row = df_names.loc[idx]
        results.append({
            "firm_name": row["firm_name_clean"],
            "firm_location": row.get("firm_loc_clean", ""),
            "company_id": company_id,
            "firm_id": anonymize_firm_id(company_id),
            "company_name": cid_to_name.get(company_id, ""),
            "match_method": method,
            "match_score": score,
            "needs_review": method in REVIEW_METHODS,
        })
        unmatched_indices.discard(idx)

    # Pass 1: Manual fixes (name + location)
    logger.info("\n--- Pass 1: Manual fixes (name + location) ---")
    n_pass1 = 0
    for idx in list(unmatched_indices):
        row = df_names.loc[idx]
        name = row["firm_name_clean"].lower()
        loc = row.get("firm_loc_clean")
        loc_lower = loc.lower() if pd.notna(loc) and loc else None

        # Try exact name+loc from manual fixes
        if loc_lower and (name, loc_lower) in mf_nameloc:
            record_match(idx, mf_nameloc[(name, loc_lower)], "manual_fix", 1.0)
            n_pass1 += 1
        elif name in mf_nameonly and len(mf_nameonly[name]) == 1:
            # Name-only manual fix (unambiguous)
            record_match(idx, list(mf_nameonly[name])[0], "manual_fix", 1.0)
            n_pass1 += 1

    logger.info(f"  Matched: {n_pass1} | Remaining: {len(unmatched_indices)}")

    # Pass 2: Exact normalized match (name + location)
    logger.info("\n--- Pass 2: Exact normalized (name + location) ---")
    n_pass2 = 0
    for idx in list(unmatched_indices):
        row = df_names.loc[idx]
        name_norm = row["name_normalized"]
        loc = row.get("firm_loc_clean")

        if name_norm in norm_index:
            entries = norm_index[name_norm]
            # If only 1 company_id, match directly
            company_ids = set(e["company_id"] for e in entries)
            if len(company_ids) == 1:
                record_match(idx, list(company_ids)[0], "exact_norm", 1.0)
                n_pass2 += 1
            elif loc and pd.notna(loc):
                # Disambiguate by location
                loc_norm = normalize_name(loc)
                loc_matches = [
                    e for e in entries
                    if e["location"] and normalize_name(str(e["location"])) == loc_norm
                ]
                loc_cids = set(e["company_id"] for e in loc_matches)
                if len(loc_cids) == 1:
                    record_match(idx, list(loc_cids)[0], "exact_norm_loc", 1.0)
                    n_pass2 += 1

    logger.info(f"  Matched: {n_pass2} | Remaining: {len(unmatched_indices)}")

    # Pass 3: Exact normalized match (name only, skip ambiguous)
    logger.info("\n--- Pass 3: Exact normalized (name only, unambiguous) ---")
    n_pass3 = 0
    for idx in list(unmatched_indices):
        row = df_names.loc[idx]
        name_norm = row["name_normalized"]

        if name_norm in norm_index:
            entries = norm_index[name_norm]
            company_ids = set(e["company_id"] for e in entries)
            if len(company_ids) == 1:
                record_match(idx, list(company_ids)[0], "exact_norm_nameonly", 1.0)
                n_pass3 += 1

    logger.info(f"  Matched: {n_pass3} | Remaining: {len(unmatched_indices)}")

    # Pass 4: Core token matching
    logger.info("\n--- Pass 4: Core token matching (all tokens of shorter in longer) ---")
    n_pass4 = 0

    # Build lookup token sets: company_id → list of (token_set, name_norm)
    company_token_sets = defaultdict(list)
    for _, row in df_lookup.iterrows():
        tokens = get_core_tokens(row["name_normalized"])
        if tokens:
            company_token_sets[int(row["company_id"])].append(
                (set(tokens), row["name_normalized"])
            )

    # For each unmatched name, check if its core tokens are a subset of any lookup name's tokens
    # OR if any lookup name's tokens are a subset of this name's tokens
    # REQUIRE: minimum 2 core tokens on both sides to avoid false positives
    for idx in list(unmatched_indices):
        row = df_names.loc[idx]
        input_tokens = set(row["name_tokens"])
        if not input_tokens or len(input_tokens) < 2:
            continue  # Need at least 2 core tokens for token matching

        loc = row.get("firm_loc_clean")
        candidates = defaultdict(float)

        # Find all lookup entries that share at least one token
        candidate_cids = set()
        for t in input_tokens:
            for cid, _ in token_index.get(t, set()):
                candidate_cids.add(cid)

        for cid in candidate_cids:
            for lookup_tokens, lookup_name in company_token_sets.get(cid, []):
                if not lookup_tokens or len(lookup_tokens) < 2:
                    continue  # Lookup entry also needs 2+ core tokens
                shorter = input_tokens if len(input_tokens) <= len(lookup_tokens) else lookup_tokens
                longer = lookup_tokens if len(input_tokens) <= len(lookup_tokens) else input_tokens

                # All tokens of shorter must appear in longer
                if shorter.issubset(longer):
                    score = len(shorter) / max(len(longer), 1)
                    candidates[cid] = max(candidates[cid], score)

        if not candidates:
            continue

        # Pick best candidate
        best_cid = max(candidates, key=candidates.get)
        best_score = candidates[best_cid]
        n_candidates = len(candidates)

        # Require minimum score of 0.5 (at least half of longer name's tokens matched)
        if best_score < 0.5:
            continue

        if n_candidates == 1 or (best_score >= 0.67):
            # Unambiguous or high-confidence (2/3+ tokens match)
            record_match(idx, best_cid, "token_exact", best_score)
            n_pass4 += 1
        elif n_candidates > 1 and loc:
            # Try location disambiguation — require score >= 0.60 to avoid
            # false positives from geographic tokens (e.g., "St. Gallen")
            loc_norm = normalize_name(str(loc))
            for cid in candidates:
                if candidates[cid] < 0.60:
                    continue  # Too low confidence for location disambiguation
                entries = [e for e in norm_index.get(row["name_normalized"], [])
                           if e["company_id"] == cid]
                if not entries:
                    # Check lookup for this company
                    cid_entries = df_lookup[df_lookup["company_id"] == cid]
                    for _, le in cid_entries.iterrows():
                        if le.get("location") and normalize_name(str(le["location"])) == loc_norm:
                            record_match(idx, cid, "token_exact_loc", candidates[cid])
                            n_pass4 += 1
                            break
                    if idx not in unmatched_indices:
                        break

    logger.info(f"  Matched: {n_pass4} | Remaining: {len(unmatched_indices)}")

    # Pass 5: Relaxed token matching (N-1 of N tokens for N≥3)
    logger.info("\n--- Pass 5: Relaxed token matching (N-1 of N tokens) ---")
    n_pass5 = 0
    for idx in list(unmatched_indices):
        row = df_names.loc[idx]
        input_tokens = set(row["name_tokens"])
        if not input_tokens or len(input_tokens) < 3:
            continue

        candidates = defaultdict(float)
        candidate_cids = set()
        for t in input_tokens:
            for cid, _ in token_index.get(t, set()):
                candidate_cids.add(cid)

        for cid in candidate_cids:
            for lookup_tokens, lookup_name in company_token_sets.get(cid, []):
                if not lookup_tokens or len(lookup_tokens) < 2:
                    continue
                shorter = input_tokens if len(input_tokens) <= len(lookup_tokens) else lookup_tokens
                longer = lookup_tokens if len(input_tokens) <= len(lookup_tokens) else input_tokens

                if len(shorter) < 3:
                    continue

                overlap = shorter & longer
                # N-1 of N tokens match
                if len(overlap) >= len(shorter) - 1:
                    score = len(overlap) / max(len(longer), 1)
                    candidates[cid] = max(candidates[cid], score)

        if not candidates:
            continue

        best_cid = max(candidates, key=candidates.get)
        best_score = candidates[best_cid]
        n_candidates = len(candidates)

        # Require minimum score of 0.5 and single candidate
        if n_candidates == 1 and best_score >= 0.5:
            record_match(idx, best_cid, "token_relaxed", best_score)
            n_pass5 += 1

    logger.info(f"  Matched: {n_pass5} | Remaining: {len(unmatched_indices)}")

    # Pass 6: Character-level fuzzy (last resort)
    logger.info("\n--- Pass 6: Character-level fuzzy (Jaro-Winkler ≥0.92) ---")
    n_pass6 = 0
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.process import extractOne

        # Build list of (name_normalized, company_id) for unambiguous entries
        fuzzy_candidates = []
        seen = set()
        for _, row in df_lookup.iterrows():
            name_norm = row["name_normalized"]
            cid = int(row["company_id"])
            if row.get("is_ambiguous"):
                continue
            key = (name_norm, cid)
            if key not in seen:
                fuzzy_candidates.append((name_norm, cid))
                seen.add(key)

        fuzzy_names = [fc[0] for fc in fuzzy_candidates]

        for idx in list(unmatched_indices):
            row = df_names.loc[idx]
            name_norm = row["name_normalized"]
            if not name_norm:
                continue

            result = extractOne(
                name_norm, fuzzy_names,
                scorer=fuzz.WRatio,
                score_cutoff=92,
            )
            if result:
                match_name, score, match_idx = result
                cid = fuzzy_candidates[match_idx][1]
                record_match(idx, cid, "fuzzy_char", score / 100.0)
                n_pass6 += 1

    except ImportError:
        logger.warning("rapidfuzz not installed, skipping Pass 6")

    logger.info(f"  Matched: {n_pass6} | Remaining: {len(unmatched_indices)}")

    # --- Save results ---
    logger.info("\n" + "=" * 60)
    logger.info("Saving results...")

    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        # Save raw matching output (untouched by LLM or manual review)
        matched_raw_path = output_dir / "matched_names_raw.csv"
        df_results.to_csv(matched_raw_path, index=False)
        logger.info(f"  Matched (raw): {len(df_results)} names → {matched_raw_path}")

        # Split review needed
        df_review = df_results[df_results["needs_review"]]
        if len(df_review) > 0:
            review_path = output_dir / "review_needed.csv"
            df_review.to_csv(review_path, index=False)
            logger.info(f"  Needs review: {len(df_review)} → {review_path}")
    else:
        logger.warning("  No matches found!")

    # Save unmatched
    if unmatched_indices:
        df_unmatched = df_names.loc[list(unmatched_indices)].copy()
        df_unmatched = df_unmatched[[name_col] + ([loc_col] if loc_col else [])]
        unmatched_path = output_dir / "unmatched_names.csv"
        df_unmatched.to_csv(unmatched_path, index=False)
        logger.info(f"  Unmatched: {len(df_unmatched)} → {unmatched_path}")

    # --- Summary ---
    total = len(df_names)
    n_matched = len(results)
    n_unmatched = len(unmatched_indices)

    logger.info("\n" + "=" * 60)
    logger.info("MATCHING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total input names:   {total}")
    logger.info(f"  Matched:             {n_matched} ({100 * n_matched / total:.1f}%)")
    logger.info(f"  Unmatched:           {n_unmatched} ({100 * n_unmatched / total:.1f}%)")
    logger.info(f"  Needs review:        {sum(1 for r in results if r['needs_review'])}")
    logger.info(f"  Unique companies:    {len(set(r['company_id'] for r in results))}")

    logger.info("\n  By method:")
    if results:
        method_counts = pd.Series([r["match_method"] for r in results]).value_counts()
        cumulative = 0
        for method, count in method_counts.items():
            cumulative += count
            logger.info(f"    {method}: {count} (+{100 * count / total:.1f}%, "
                        f"cumul: {100 * cumulative / total:.1f}%)")

    # Sample matches for each method
    if results:
        logger.info("\n  Sample matches (5 per method):")
        df_res = pd.DataFrame(results)
        for method in df_res["match_method"].unique():
            subset = df_res[df_res["match_method"] == method].head(5)
            logger.info(f"\n    [{method}]:")
            for _, r in subset.iterrows():
                logger.info(f"      '{r['firm_name']}' → {r['company_name']} "
                            f"(cid={r['company_id']}, score={r['match_score']:.2f})")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 complete.")
    logger.info("=" * 60)


# ============================================================
# Phase 3: LLM Review of Fuzzy Matches
# ============================================================

LLM_REVIEW_SYSTEM_PROMPT = """You are reviewing automated employer name matches from a Swiss survey dataset.

Context: Survey interviewers in Switzerland hand-enter employer names (in German, French, or Italian). An algorithm matched these to companies in a business registry. You must verify whether each match is correct.

For each numbered pair, determine if the survey employer name refers to the SAME company as the matched registry name. Consider:
- Department/building suffixes are fine (e.g., "Credit Suisse AG Private Banking" = "Credit Suisse AG")
- Regional variants of the same company are fine (e.g., "Migros Aare M Solothurn" = "Genossenschaft Migros Aare")
- Different companies that happen to share a word are NOT matches (e.g., "Sportamt St. Gallen" ≠ "Universität St. Gallen")
- A person's name is NOT a company match unless it's clearly the same entity

Reply with ONLY a JSON array of objects, one per pair:
[{"id": 1, "verdict": "YES"}, {"id": 2, "verdict": "NO"}, ...]

verdict must be exactly one of: "YES", "NO", "UNCERTAIN"
"""


def llm_review(args) -> None:
    """Use LLM to review fuzzy/token matches for correctness."""
    import json
    import time

    try:
        import openai
        from dotenv import load_dotenv
    except ImportError:
        raise ImportError("openai and python-dotenv packages required. Install with: pip install openai python-dotenv")

    load_dotenv("config.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in config.env")

    client = openai.OpenAI(api_key=api_key, timeout=120.0)
    model = args.model

    # Load review file
    review_path = Path(args.review_file)
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}. Run match-names first.")
    df_review = pd.read_csv(review_path)
    logger.info(f"Loaded {len(df_review)} matches to review")
    logger.info(f"Model: {model}, Batch size: {args.batch_size}")

    # Check for existing progress (checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "llm_review_checkpoint.csv"

    reviewed_keys = set()
    all_verdicts = []
    if checkpoint_path.exists():
        df_checkpoint = pd.read_csv(checkpoint_path)
        reviewed_keys = set(zip(df_checkpoint["firm_name"], df_checkpoint["firm_location"].fillna("")))
        all_verdicts = df_checkpoint.to_dict("records")
        logger.info(f"Resuming from checkpoint: {len(reviewed_keys)} already reviewed")

    # Filter to only unreviewed
    remaining = df_review[
        ~df_review.apply(lambda r: (r["firm_name"], str(r.get("firm_location", "") or "")), axis=1).isin(reviewed_keys)
    ]
    logger.info(f"Remaining to review: {len(remaining)}")

    if len(remaining) == 0:
        logger.info("All matches already reviewed!")
    else:
        # Process in batches
        batch_size = args.batch_size
        batches = [remaining.iloc[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
        total_batches = len(batches)
        total_input_tokens = 0
        total_output_tokens = 0

        logger.info(f"Processing {total_batches} batches of up to {batch_size} pairs each...")

        for batch_num, batch in enumerate(batches, 1):
            # Build prompt with numbered pairs
            pairs_text = []
            batch_records = []
            for pair_id, (idx, row) in enumerate(batch.iterrows(), 1):
                loc_str = f" (location: {row['firm_location']})" if pd.notna(row.get("firm_location")) and str(row.get("firm_location")).strip() else ""
                pairs_text.append(
                    f"{pair_id}. Survey name: \"{row['firm_name']}\"{loc_str}\n"
                    f"   Matched to: \"{row['company_name']}\" (company_id={row['company_id']})\n"
                    f"   Method: {row['match_method']}, Score: {row['match_score']:.2f}"
                )
                batch_records.append({"idx": idx, "pair_id": pair_id, "row": row})

            user_prompt = "Review these employer name matches:\n\n" + "\n\n".join(pairs_text)

            # Call API with exponential backoff retry
            try:
                content = ""
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": LLM_REVIEW_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_completion_tokens=2000,
                        )
                        content = response.choices[0].message.content
                        content = content.strip() if content else ""
                        if content:
                            break
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt + random.random()
                            logger.warning(f"  Batch {batch_num}: Empty response, retrying in {wait:.1f}s...")
                            time.sleep(wait)
                    except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as retry_err:
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt + random.random()
                            logger.warning(f"  Batch {batch_num}: {type(retry_err).__name__}, retrying in {wait:.1f}s...")
                            time.sleep(wait)
                        else:
                            raise
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens

                # Parse JSON response
                # Handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]

                verdicts = json.loads(content)
                verdict_map = {v["id"]: v["verdict"] for v in verdicts}

                # Record results
                for rec in batch_records:
                    verdict = verdict_map.get(rec["pair_id"], "UNCERTAIN")
                    row = rec["row"]
                    all_verdicts.append({
                        "firm_name": row["firm_name"],
                        "firm_location": row.get("firm_location", ""),
                        "company_id": row["company_id"],
                        "firm_id": row.get("firm_id", ""),
                        "company_name": row["company_name"],
                        "match_method": row["match_method"],
                        "match_score": row["match_score"],
                        "llm_verdict": verdict,
                    })

                n_yes = sum(1 for v in verdicts if v.get("verdict") == "YES")
                n_no = sum(1 for v in verdicts if v.get("verdict") == "NO")
                n_unc = sum(1 for v in verdicts if v.get("verdict") == "UNCERTAIN")
                logger.info(f"  Batch {batch_num}/{total_batches}: "
                            f"YES={n_yes}, NO={n_no}, UNCERTAIN={n_unc}")

            except json.JSONDecodeError as e:
                logger.warning(f"  Batch {batch_num}: Failed to parse JSON response: {e}")
                logger.warning(f"  Raw response: {content[:200]}")
                # Mark all as UNCERTAIN
                for rec in batch_records:
                    row = rec["row"]
                    all_verdicts.append({
                        "firm_name": row["firm_name"],
                        "firm_location": row.get("firm_location", ""),
                        "company_id": row["company_id"],
                        "firm_id": row.get("firm_id", ""),
                        "company_name": row["company_name"],
                        "match_method": row["match_method"],
                        "match_score": row["match_score"],
                        "llm_verdict": "PARSE_ERROR",
                    })

            except openai.APIError as e:
                logger.warning(f"  Batch {batch_num}: API error: {e}")
                for rec in batch_records:
                    row = rec["row"]
                    all_verdicts.append({
                        "firm_name": row["firm_name"],
                        "firm_location": row.get("firm_location", ""),
                        "company_id": row["company_id"],
                        "firm_id": row.get("firm_id", ""),
                        "company_name": row["company_name"],
                        "match_method": row["match_method"],
                        "match_score": row["match_score"],
                        "llm_verdict": "API_ERROR",
                    })

            # Save checkpoint after each batch
            pd.DataFrame(all_verdicts).to_csv(checkpoint_path, index=False)

            # Brief pause between batches to avoid rate limits
            if batch_num < total_batches:
                time.sleep(0.5)

        # Cost summary
        # GPT-5-mini pricing: $0.25/1M input, $2.00/1M output
        pricing = {
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
            "gpt-5": {"input": 1.25, "output": 10.00},
        }
        p = pricing.get(model, pricing["gpt-5-mini"])
        cost = (total_input_tokens * p["input"] + total_output_tokens * p["output"]) / 1_000_000
        logger.info(f"\nAPI usage: {total_input_tokens} input + {total_output_tokens} output tokens")
        logger.info(f"Estimated cost: ${cost:.4f}")

    # --- Save final results ---
    df_verdicts = pd.DataFrame(all_verdicts)

    # Split by verdict
    df_confirmed = df_verdicts[df_verdicts["llm_verdict"] == "YES"]
    df_rejected = df_verdicts[df_verdicts["llm_verdict"] == "NO"]
    df_uncertain = df_verdicts[df_verdicts["llm_verdict"].isin(["UNCERTAIN", "PARSE_ERROR", "API_ERROR"])]

    confirmed_path = output_dir / "llm_confirmed.csv"
    rejected_path = output_dir / "llm_rejected.csv"
    uncertain_path = output_dir / "llm_uncertain.csv"
    full_path = output_dir / "llm_review_full.csv"

    df_confirmed.to_csv(confirmed_path, index=False)
    df_rejected.to_csv(rejected_path, index=False)
    if len(df_uncertain) > 0:
        df_uncertain.to_csv(uncertain_path, index=False)
    df_verdicts.to_csv(full_path, index=False)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info("\n" + "=" * 60)
    logger.info("LLM REVIEW SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total reviewed:    {len(df_verdicts)}")
    logger.info(f"  Confirmed (YES):   {len(df_confirmed)} → {confirmed_path}")
    logger.info(f"  Rejected (NO):     {len(df_rejected)} → {rejected_path}")
    logger.info(f"  Uncertain:         {len(df_uncertain)} → {uncertain_path}")

    # Breakdown by match method
    logger.info("\n  By match method:")
    for method in df_verdicts["match_method"].unique():
        subset = df_verdicts[df_verdicts["match_method"] == method]
        n_y = (subset["llm_verdict"] == "YES").sum()
        n_n = (subset["llm_verdict"] == "NO").sum()
        n_u = subset["llm_verdict"].isin(["UNCERTAIN", "PARSE_ERROR", "API_ERROR"]).sum()
        logger.info(f"    {method}: {len(subset)} total → YES={n_y}, NO={n_n}, UNCERTAIN={n_u}")

    logger.info("\n  LLM review outputs saved. Run 'compile' to produce matched_names_final.csv.")
    logger.info("\n" + "=" * 60)
    logger.info("LLM review complete.")
    logger.info("=" * 60)


# ============================================================
# Compile: Combine raw matches + LLM verdicts + manual verdicts
# ============================================================

def compile_results(args):
    """Combine raw matching output with LLM and manual review verdicts.

    Pipeline:
      matched_names_raw.csv  (Phase 2 output)
      - llm_rejected.csv     (remove these)
      - manual_verdicts.csv NO entries (remove these)
      + llm_confirmed.csv    (set needs_review=False)
      + manual_verdicts.csv YES entries (set needs_review=False)
      = matched_names_final.csv
    """
    output_dir = Path(args.output_dir)

    # --- Load raw matches ---
    raw_path = output_dir / "matched_names_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw matches not found: {raw_path}\nRun 'match-names' first.")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} raw matches from {raw_path.name}")
    initial_count = len(df)

    def make_key(row):
        return (str(row.get("firm_name", "")), str(row.get("firm_location", "")), str(row.get("company_id", "")))

    # --- Load manual verdicts first (they override LLM) ---
    manual_path = output_dir / "manual_verdicts.csv"
    manual_yes_keys = set()
    manual_no_keys = set()
    if manual_path.exists():
        df_manual = pd.read_csv(manual_path)
        manual_yes_keys = set(
            (str(r["firm_name"]), str(r.get("firm_location", "")), str(r["company_id"]))
            for _, r in df_manual[df_manual["verdict"] == "YES"].iterrows()
        )
        manual_no_keys = set(
            (str(r["firm_name"]), str(r.get("firm_location", "")), str(r["company_id"]))
            for _, r in df_manual[df_manual["verdict"] == "NO"].iterrows()
        )

    # --- Apply LLM rejections (skip entries manually confirmed as YES) ---
    llm_rejected_path = output_dir / "llm_rejected.csv"
    n_llm_rejected = 0
    if llm_rejected_path.exists():
        df_rej = pd.read_csv(llm_rejected_path)
        rejected_keys = set(
            (str(r["firm_name"]), str(r.get("firm_location", "")), str(r["company_id"]))
            for _, r in df_rej.iterrows()
        ) - manual_yes_keys  # Manual YES overrides LLM NO
        mask = df.apply(lambda r: make_key(r) not in rejected_keys, axis=1)
        n_llm_rejected = len(df) - mask.sum()
        df = df[mask].copy()
        logger.info(f"  LLM rejected: removed {n_llm_rejected} matches")
    else:
        logger.info(f"  No LLM rejected file found (skipping)")

    # --- Apply manual verdict rejections ---
    n_manual_rejected = 0
    n_manual_confirmed = 0
    if manual_no_keys:
        mask = df.apply(lambda r: make_key(r) not in manual_no_keys, axis=1)
        n_manual_rejected = len(df) - mask.sum()
        df = df[mask].copy()
    if manual_yes_keys:
        df["needs_review"] = df.apply(
            lambda r: False if make_key(r) in manual_yes_keys else r["needs_review"],
            axis=1,
        )
        n_manual_confirmed = len(manual_yes_keys)
    if manual_path.exists():
        logger.info(f"  Manual verdicts: removed {n_manual_rejected}, confirmed {n_manual_confirmed}")
    else:
        logger.info(f"  No manual verdicts file found (skipping)")

    # --- Apply LLM confirmations ---
    llm_confirmed_path = output_dir / "llm_confirmed.csv"
    n_llm_confirmed = 0
    if llm_confirmed_path.exists():
        df_conf = pd.read_csv(llm_confirmed_path)
        confirmed_keys = set(
            (str(r["firm_name"]), str(r.get("firm_location", "")), str(r["company_id"]))
            for _, r in df_conf.iterrows()
        )
        df["needs_review"] = df.apply(
            lambda r: False if make_key(r) in confirmed_keys else r["needs_review"],
            axis=1,
        )
        n_llm_confirmed = len(df_conf)
        logger.info(f"  LLM confirmed: {n_llm_confirmed} matches (needs_review → False)")
    else:
        logger.info(f"  No LLM confirmed file found (skipping)")

    # --- Flag borderline matches as needs_review even if LLM confirmed ---
    # fuzzy_char and low-score token matches are inherently less reliable
    borderline_mask = (
        (df["match_method"] == "fuzzy_char") |
        ((df["match_method"].isin(["token_exact", "token_relaxed"])) & (df["match_score"] <= 0.50))
    )
    n_borderline_flagged = ((borderline_mask) & (~df["needs_review"])).sum()
    df.loc[borderline_mask, "needs_review"] = True
    logger.info(f"  Borderline flagged: {n_borderline_flagged} matches set to needs_review=True")

    # --- Save final output ---
    final_path = output_dir / "matched_names_final.csv"
    df.to_csv(final_path, index=False)

    # --- Summary ---
    n_review = df["needs_review"].sum()
    n_confirmed = len(df) - n_review
    logger.info("\n" + "=" * 60)
    logger.info("COMPILE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Raw matches:          {initial_count}")
    logger.info(f"  LLM rejected:        -{n_llm_rejected}")
    logger.info(f"  Manual rejected:     -{n_manual_rejected}")
    logger.info(f"  Final matched:        {len(df)}")
    logger.info(f"    Confirmed:          {n_confirmed}")
    logger.info(f"    Still needs review: {n_review}")
    logger.info(f"  Output: {final_path}")
    logger.info("=" * 60)

    # Traceability summary
    logger.info("\n  Input files used:")
    logger.info(f"    {raw_path.name} ({initial_count} entries)")
    if llm_rejected_path.exists():
        logger.info(f"    {llm_rejected_path.name} ({n_llm_rejected} removals)")
    if llm_confirmed_path.exists():
        logger.info(f"    {llm_confirmed_path.name} ({n_llm_confirmed} confirmations)")
    if manual_path.exists():
        logger.info(f"    {manual_path.name} ({n_manual_rejected} removals, {n_manual_confirmed} confirmations)")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Improve SHP-to-X28 firm matching coverage"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Build lookup
    p_build = subparsers.add_parser("build-lookup",
                                     help="Build enhanced lookup from all data sources")
    p_build.add_argument("--scrape-file",
                         default=str(COAUTHOR_DATA_DIR / "created" / "scrape_uid_complete.csv"))
    p_build.add_argument("--uid-final-file",
                         default=str(COAUTHOR_DATA_DIR / "created" / "shp_uid_final.csv"))
    p_build.add_argument("--x28-match-file",
                         default=str(COAUTHOR_DATA_DIR / "created" / "shp_x28_match.csv"))
    p_build.add_argument("--company-mapping",
                         default=str(DATA_DIR / "company_mapping.csv"))
    p_build.add_argument("--output-dir",
                         default=str(DATA_DIR / "shp_matching"))

    # Match names
    p_match = subparsers.add_parser("match-names",
                                     help="Match firm names against enhanced lookup")
    p_match.add_argument("--lookup-file",
                         default=str(DATA_DIR / "shp_matching" / "enhanced_lookup.csv"))
    p_match.add_argument("--manual-fixes-file",
                         default=str(DATA_DIR / "shp_matching" / "manual_fixes.csv"))
    p_match.add_argument("--names-file", required=True,
                         help="CSV with firm_name (and optional firm_location) columns")
    p_match.add_argument("--output-dir",
                         default=str(DATA_DIR / "shp_matching"))

    # LLM review
    p_review = subparsers.add_parser("llm-review",
                                      help="Use LLM to review fuzzy/token matches")
    p_review.add_argument("--review-file",
                           default=str(DATA_DIR / "shp_matching" / "review_needed.csv"),
                           help="CSV of matches to review (from match-names)")
    p_review.add_argument("--output-dir",
                           default=str(DATA_DIR / "shp_matching"))
    p_review.add_argument("--model", default="gpt-5-mini",
                           help="OpenAI model for review (default: gpt-5-mini)")
    p_review.add_argument("--batch-size", type=int, default=20,
                           help="Number of pairs per API call (default: 20)")

    # Compile
    p_compile = subparsers.add_parser("compile",
                                       help="Combine raw matches + LLM/manual verdicts → final output")
    p_compile.add_argument("--output-dir",
                            default=str(DATA_DIR / "shp_matching"))

    args = parser.parse_args()

    if args.mode == "build-lookup":
        build_lookup(args)
    elif args.mode == "match-names":
        match_names(args)
    elif args.mode == "llm-review":
        llm_review(args)
    elif args.mode == "compile":
        compile_results(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
