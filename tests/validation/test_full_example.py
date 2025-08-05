#!/usr/bin/env python3
"""
Test the conservative cleaning function with the full problematic example
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conservative_cleaning import clean_job_ad_conservative

def test_full_example():
    # The problematic example from the user
    test_text = """Senior AI Strategy Consulting

Wir sind eine Schweizer IT-Consulting-Firma. Wir bauen innovative Lösungen bei unseren Kunden vor Ort. Uns gibt es seit über 25 Jahren, aber wir leben immer noch eine Start-up-Mentalität – 170 Leute, Flat Hierarchy, No Politics, Lots of Fun! Bei uns gilt: People First! Alle können sich bei uns einbringen und zu Expertinnen und Experten werden. Wissen wird geteilt und gegenseitige Unterstützung ist zentral. We're ONE Team!
Bewirb Dich jetzt!
Deine Challenges

Du berätst unsere Kunden und entwickelst mit ihnen eine AI Strategie. Dabei diskutierst Du den Impact von AI auf das Unternehmen, weist auf Chancen und Gefahren in der Branche hin und entwickelst Szenarien für den Einsatz von AI

Du entwickelst mit den Fachabteilungen zusammen neue AI Use Cases in Workshops

Nutzen Analyse, Bewertung und Priorisierung von AI Use Cases

Definieren der Zielarchitektur und dem Vorgehen für die Umsetzung

In der Umsetzung der AI Strategie arbeitest Du Hand in Hand und auf Augenhöhe mit z.B. Fachvertretern und dem Analytics Team zusammen

Das bringst Du mit

Expertise im Bereich Strategieentwicklung und –beratung

Know-how in den Bereichen Artificial Intelligence, Advanced Analytics, Machine Learning, Big Data, Business Intelligence o.ä.

Abgeschlossenes Studium (Informatik, Wirtschaftsinformatik, Mathematik, Statistik, Elektrotechnik, Physik o.ä.)

Stilsicherer Umgang mit C-Level und Fachverantwortlichen

Leidenschaft, sich schnell in neue Themen und Technologien einzuarbeiten

Teamplayer und gute kommunikative Fähigkeiten

Fliessende Deutsch- und Englischkenntnisse

Das sind unsere Benefits für Dich

Wissensaustausch untereinander: Monatliche Mitarbeitendenevents, TechBiers und Scrum'n'Wines garantieren Know-How Austausch und frische Ideen

Life-Balance: Jahresarbeitszeit, Überzeitkompensation, 5 Wochen Ferien, die Zuger Feiertage und Teilzeitangebote – denn der Ausgleich ist uns wichtig

Garantierte Abwechslung: Verschiedene Kunden, Themen, Methoden und Technologien

Hardfacts

Dein Arbeitsort ist vor Ort beim Kunden in der Deutschschweiz (Zürich, Luzern, Basel, Bern, St.Gallen)

Pensum: 80% oder mehr

Dieses Stelleninserat umschreibt ein Szenario, das Dich in einem Projekt erwarten könnte. Wir suchen Talente, die für verschiedene Themen und Technologien offen sind.

Hast Du Fragen?

Michaela Kuhn
People & Development + Associate Partner

Wir freuen uns auf Deine Bewerbung!
Anrede Herr Frau Vorname Name E-Mail Telefon Dein Xing-Profil Dein Linkedin-Profil Bewerbungsschreiben

.pdf, .doc, .docx, .zip (max. 20MB)

Lebenslauf

.pdf, .doc, .docx, .zip (max. 20MB)

Weitere Dokumente

.pdf, .doc, .docx, .zip (max. 20MB)

Weitere Links Bemerkung Wie bist Du auf ipt aufmerksam geworden? Ich habe die Datenschutzbestimmungen gelesen und bin damit einverstanden. Comments"""

    print("ORIGINAL TEXT:")
    print("=" * 60)
    print(test_text)
    print("\n" + "=" * 60)
    print("CONSERVATIVE CLEANING RESULT:")
    print("=" * 60)
    
    cleaned = clean_job_ad_conservative(test_text)
    print(cleaned)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Original length: {len(test_text)} characters")
    print(f"Cleaned length: {len(cleaned)} characters")
    print(f"Reduction: {((len(test_text) - len(cleaned)) / len(test_text) * 100):.1f}%")

if __name__ == "__main__":
    test_full_example()