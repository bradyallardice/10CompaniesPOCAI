import pandas as pd
import numpy as np
import json
import psycopg2
import random
import os
from dotenv import load_dotenv

def load_company_ads(num_companies=100, output_path=None, exclude_companies=None, dataset_type="train"):
    """
    Connect to the PostgreSQL database and retrieve job postings
    from a random selection of companies. Save the results to a CSV file.
    
    Parameters:
    -----------
    num_companies : int, optional
        Number of random companies to select (default: 100)
    output_path : str, optional
        Path where to save the CSV file. If None, a default name will be used.
    exclude_companies : list, optional
        List of company names to exclude from selection
    dataset_type : str, optional
        Type of dataset ("train" or "test") for filename
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing job ads from randomly selected companies
    """
    # Ensure num_companies is an integer
    num_companies = int(num_companies) if not isinstance(num_companies, int) else num_companies
    
    # Load environment variables
    load_dotenv('config.env')
    
    # Database connection details from environment
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    
    cur = conn.cursor()
    
    try:
        # Get all distinct company IDs
        cur.execute("""
            SELECT DISTINCT company_id
            FROM job_postings;
        """)
        
        all_company_ids = [row[0] for row in cur.fetchall()]
        
        # Select random company IDs
        if len(all_company_ids) <= num_companies:
            random_company_ids = all_company_ids
            print(f"Warning: Only {len(all_company_ids)} unique company IDs found, using all of them.")
        else:
            random_company_ids = random.sample(all_company_ids, num_companies)
        
        # Query to select rows where company_id is in the list of random IDs
        query = """
            SELECT *
            FROM job_postings
            WHERE company_id = ANY(%s);
        """
        
        cur.execute(query, (random_company_ids,))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        
        # Create DataFrame
        df_random_ads = pd.DataFrame(rows, columns=colnames)
        
        # Save to CSV  
        df_random_ads.to_csv(f'{output_path}random_sample_{num_companies}_companies_ads.csv', index=False)
        print(f"Random company job ads saved to {output_path}random_sample_{num_companies}_companies_ads.csv")
        
        return df_random_ads
        
    finally:
        # Close cursor and connection
        cur.close()
        conn.close()

def create_test_set(num_companies=50, output_path=None):
    """
    Create a test set by selecting companies that are NOT in the original 100 company training set.
    
    Parameters:
    -----------
    num_companies : int, optional
        Number of random companies to select for test set (default: 50)
    output_path : str, optional
        Path where to save the CSV file. If None, uses "Data/"
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing job ads from test set companies
    """
    # All 100 companies from the original training set that need to be excluded
    training_companies = [
        "A. Köhler AG", "Akeret Baumanagement AG", "Aktiengesellschaft Cilander", 
        "Aletsch Bahnen AG", "Alters- und Gesundheitszentrum (AGZ)", "Appenzeller Verlag AG",
        "Arthouse Commercio Movie AG", "Assura Holding SA", "Au Bonheur des Animaux Myriam Lienhard",
        "Berner Bildungszentrum Pflege AG", "Bison Schweiz AG", "Bourgeois Avocats SA",
        "Bourquin SA", "Bundesamt für Zivilluftfahrt (BAZL)", "Bäckerei Konditorei Zeller AG",
        "CHEMODEX AG", "CKW AG", "Cecchettin SA", "Commune d'Arzier-Le Muids",
        "Confiserie Eichenberger AG", "Cybersystems GmbH", "Eidgenössisches Justiz- und Polizeidepartement (EJPD)",
        "Energie Wasser Bern", "FNX Sàrl", "Franke Industrie AG", "Fun Planet Loisirs Brig-Glis AG",
        "Gate Gourmet Switzerland GmbH", "Gebr. Marthaler AG", "Gebro Pharma AG", "Gemeinde Grabs",
        "Gemeinde Matten", "Gemeinde Zollikon (ZH)", "Genossenschaft Lindenmühle", "Gloor Metallbau GmbH",
        "Grünig-Interscreen AG", "Hammer Metall AG", "Häfliger und Partner AG", "IM Architektur AG",
        "ISP Electro Solutions AG", "Imoberdorf AG", "Innovation Process Technology AG", "Ivoclar Vivadent AG",
        "Jaisli-Xamax AG", "John Schwab S.A.", "KV Zürich Business School", "KiK Kultur im Kammgarn",
        "Kita Kiddi 2", "Knuchel Farben AG", "Küng Rechtsanwälte & Notare AG", "Lichtensteiger AG Bäckerei",
        "Manufacture Jaeger-LeCoultre, Branch of Richemont International SA", "Metzgerei Kast GmbH",
        "Mivelaz Bois SA", "Mändli Handels- und Montage AG", "Narcocare AG", "North Thin Ply Technology Sàrl",
        "Obergericht des Kantons Zürich", "Optiprint AG", "PPCmetrics AG", "Perlen Packaging AG, Perlen",
        "Police Nyon Région", "Qualipet AG", "R.I.C. Risk & Insurance Consulting AG", "Region Oberaargau",
        "Restaurant Bauernhof AG", "Resto-Lounge Sàrl", "Rudolf Wirz Strassen- und Tiefbau AG",
        "SCHMID WETLI AG", "SIB Schweiz. Institut für Betriebsökonomie AG", "STB Engineering AG",
        "Schneeberger Décolletages S.A.", "Schweizerische Agentur für Innovationsförderung (Innosuisse)",
        "Schweizerisches Rotes Kreuz Kanton Schaffhausen", "Sedelec SA Lausanne", "Selmoni Ingenieur AG",
        "Senevita Mülibach AG", "Seniorenzentrum Zwyden", "Sepp Knüsel AG", "SmartLiberty SA",
        "Société coopérative Générations", "Spirig HealthCare AG", 
        "Spitex für Stadt und Land AG, Zweigniederlassung Schwyz", "Spühler Partner Architekten AG",
        "Stiftung AR SUNNSYTE Wohnen begleiten pflegen", "Stiftung Besuchsdienst Innerschweiz BDI",
        "Tamedia Espace AG", "Th. Willy AG Auto-Zentrum", "Triag AG", "Trisa Accessoires AG",
        "Union des Associations Européennes de Football (UEFA)", "Verein Spitex Heitersberg",
        "Verein pflegimuri", "Walter Lüthi Holzbau AG", "Windredli GmbH", "Zürcher Kunstgesellschaft",
        "hauswartprofis AG", "inklusia", "shelterschweiz", "suter & gerteis AG", "säntis packaging ag"
    ]
    
    # Ensure num_companies is an integer
    num_companies = int(num_companies) if not isinstance(num_companies, int) else num_companies
    
    # Set default output path
    if output_path is None:
        output_path = "Data/"
    
    # Load environment variables
    load_dotenv('config.env')
    
    # Database connection details from environment
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    
    cur = conn.cursor()
    
    try:
        print(f"Excluding {len(training_companies)} training companies from test set selection")
        
        # Get company_ids to exclude by matching company names
        exclude_query = """
            SELECT DISTINCT company_id 
            FROM job_postings 
            WHERE company_name = ANY(%s);
        """
        cur.execute(exclude_query, (training_companies,))
        exclude_company_ids = [row[0] for row in cur.fetchall()]
        print(f"Found {len(exclude_company_ids)} company IDs to exclude")
        
        # Get all company IDs except the excluded ones
        cur.execute("""
            SELECT DISTINCT company_id
            FROM job_postings
            WHERE company_id <> ALL(%s);
        """, (exclude_company_ids,))
        
        available_company_ids = [row[0] for row in cur.fetchall()]
        print(f"Found {len(available_company_ids)} available companies for test set")
        
        # Select random company IDs for test set
        if len(available_company_ids) <= num_companies:
            random_company_ids = available_company_ids
            print(f"Warning: Only {len(available_company_ids)} unique company IDs available, using all of them.")
        else:
            random_company_ids = random.sample(available_company_ids, num_companies)
            print(f"Selected {len(random_company_ids)} random companies for test set")
        
        # Query to select rows where company_id is in the list of random IDs
        query = """
            SELECT *
            FROM job_postings
            WHERE company_id = ANY(%s);
        """
        
        cur.execute(query, (random_company_ids,))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        
        # Create DataFrame
        df_test_ads = pd.DataFrame(rows, columns=colnames)
        
        # Save to CSV with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{output_path}test_set_{num_companies}_companies_{timestamp}.csv'
        df_test_ads.to_csv(filename, index=False)
        print(f"Test set saved to {filename}")
        
        return df_test_ads
        
    finally:
        # Close cursor and connection
        cur.close()
        conn.close()

# Example usage if run directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-test-set":
        # Create test set of 50 companies
        df_test = create_test_set(num_companies=50)
        print(f"Created test set with {len(df_test)} job ads from 50 companies (excluding original 100 training companies)")
    else:
        # Original training set creation (unchanged)
        df_random_ads = load_company_ads(num_companies=100, output_path="random_company_ads")
        print(f"Retrieved {len(df_random_ads)} job ads from 100 random companies")