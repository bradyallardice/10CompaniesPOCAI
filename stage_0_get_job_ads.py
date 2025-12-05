import pandas as pd
import numpy as np
import json
import psycopg2
import random
import os
import re
import csv
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Tuple

def normalize_text_for_matching(text):
    """
    Normalize text for matching to match database content_norm behavior exactly.
    Database content_norm only does:
    1. Converts accented characters to their base form (é -> e, ñ -> n, etc.)
    2. Converts to lowercase
    3. Cleans up whitespace
    NOTE: Database does NOT remove punctuation, so we shouldn't either
    """
    if text is None:
        return None
    
    # Remove accents (NFD normalization + ASCII encoding) - matches database
    normalized = unicodedata.normalize('NFD', str(text)).encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase and clean up whitespace - matches database
    normalized = normalized.lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def _punct_between_letters_regex(token: str) -> str:
    """
    Build a regex pattern that matches the token when written with
    punctuation/dashes between letters, e.g. 'a.i.' or 'a-i' for 'ai'.
    - Anchored to word boundaries excluding underscores
    - Allows DOT or ASCII hyphen or Unicode dashes between letters
    - Does NOT allow underscore (so 'a_i_' won't match)
    """
    assert 2 <= len(token) <= 5, f"Token '{token}' must be 2-5 characters long"
    letters = list(token)
    dashclass = r"[.\-\u2010-\u2015]\s*"

    if len(letters) == 2:
        core = rf"{re.escape(letters[0])}{dashclass}{re.escape(letters[1])}"
    elif len(letters) == 3:
        core = (
            rf"{re.escape(letters[0])}{dashclass}"
            rf"{re.escape(letters[1])}(?:{dashclass})?"
            rf"{re.escape(letters[2])}"
        )
    elif len(letters) == 4:
        core = (
            rf"{re.escape(letters[0])}{dashclass}"
            rf"{re.escape(letters[1])}{dashclass}"
            rf"{re.escape(letters[2])}{dashclass}"
            rf"{re.escape(letters[3])}"
        )
    else:  # len == 5
        core = (
            rf"{re.escape(letters[0])}{dashclass}"
            rf"{re.escape(letters[1])}{dashclass}"
            rf"{re.escape(letters[2])}{dashclass}"
            rf"{re.escape(letters[3])}{dashclass}"
            rf"{re.escape(letters[4])}"
        )
    # Word-ish boundaries that exclude underscore (Python compatible)
    return rf"(^|[^a-zA-Z0-9_]){core}([^a-zA-Z0-9_]|$)"

def load_multilingual_keywords():
    """
    Load multilingual AI keywords from CSV file.
    
    Returns:
        list: Deduplicated list of all keywords from all languages, normalized for matching
    """
    keywords = set()  # Use set to avoid duplicates
    
    # Path to the multilingual keywords file
    csv_path = Path(__file__).parent / "Data" / "ai_keywords_multilingual_v3.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Multilingual keywords file not found: {csv_path}")
    
    # Load CSV and extract all keywords
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Extract keywords from each language column (English, French, German, Italian)
            for lang_col in ['English', 'French', 'German', 'Italian']:
                if lang_col in row and row[lang_col].strip():
                    keyword = row[lang_col].strip()
                    # Add both original and normalized forms
                    keywords.add(keyword.lower())
                    keywords.add(normalize_text_for_matching(keyword))
    
    # Convert to sorted list for consistency
    keyword_list = sorted(list(keywords))
    
    print(f"Loaded {len(keyword_list)} unique multilingual AI keywords from CSV")
    return keyword_list

# Training companies list (used across multiple functions)
TRAINING_COMPANIES = [
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
            FROM job_postings_unified;
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
            FROM job_postings_unified
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
    # Use the global training companies list
    training_companies = TRAINING_COMPANIES
    
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
            FROM job_postings_unified 
            WHERE company_name = ANY(%s);
        """
        cur.execute(exclude_query, (training_companies,))
        exclude_company_ids = [row[0] for row in cur.fetchall()]
        print(f"Found {len(exclude_company_ids)} company IDs to exclude")
        
        # Get all company IDs except the excluded ones
        cur.execute("""
            SELECT DISTINCT company_id
            FROM job_postings_unified
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
            FROM job_postings_unified
            WHERE company_id = ANY(%s);
        """
        
        cur.execute(query, (random_company_ids,))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        
        # Create DataFrame
        df_test_ads = pd.DataFrame(rows, columns=colnames)
        
        # Save to CSV
        from datetime import datetime
        filename = f'{output_path}test_set_{num_companies}_companies.csv'
        df_test_ads.to_csv(filename, index=False)
        print(f"Test set saved to {filename}")
        
        return df_test_ads
        
    finally:
        # Close cursor and connection
        cur.close()
        conn.close()

def extract_keyword_matching_jobs(output_path="Data/", batch_size=50000, limit_rows=None, test_companies_only=False):
    """
    Extract all job ads from the full database that match AI development keywords.
    Uses PostgreSQL pattern matching for efficient search across 11M rows.
    
    Parameters:
    -----------
    output_path : str, optional
        Directory path where to save the CSV file (default: "Data/")
    batch_size : int, optional
        Number of rows to process in each batch (default: 50000)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all job ads that match AI keywords
    """
    
    print("="*60)
    print("EXTRACTING AI KEYWORD MATCHING JOBS FROM FULL DATABASE")
    print("="*60)
    
    # Load multilingual AI keywords from CSV file
    ai_development_keywords = load_multilingual_keywords()
    # Convert keywords to SQL patterns and group by strategy
    keyword_patterns = _prepare_sql_patterns(ai_development_keywords)
    
    # Load environment variables
    load_dotenv('config.env')
    
    # Database connection details
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
    
    try:
        print(f"Connected to database: {db_name}")
        print(f"Processing keywords: {len(ai_development_keywords)} total")
        print(f"Starting keyword pattern preparation...")
        
        # Execute keyword matching query
        print(f"Executing keyword search on {13785774:,} total job postings...")
        matching_jobs_df = _execute_keyword_search(conn, keyword_patterns, batch_size, limit_rows, test_companies_only)
        
        if matching_jobs_df is not None and len(matching_jobs_df) > 0:
            # Save results
            filename = f'{output_path}full_job_ads.csv'
            
            matching_jobs_df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f"\n{'='*60}")
            print("KEYWORD EXTRACTION COMPLETE")
            print(f"{'='*60}")
            print(f"✅ Total matching jobs: {len(matching_jobs_df):,}")
            print(f"✅ Unique companies: {matching_jobs_df['company_name'].nunique():,}")
            print(f"✅ Results saved to: {filename}")
            
            # Show top companies by job count
            print(f"\nTop 10 companies by AI job count:")
            top_companies = matching_jobs_df['company_name'].value_counts().head(10)
            for company, count in top_companies.items():
                print(f"  {company}: {count}")
                
            return matching_jobs_df
        else:
            print("❌ No matching jobs found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error during keyword extraction: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()
        print("Database connection closed")

def _prepare_sql_patterns(keywords: List[str]) -> dict:
    """
    Convert Python keywords to PostgreSQL patterns matching stage_1 logic exactly.
    Uses proper word boundaries and categorization to avoid false positives.
    """
    
    patterns = {
        'short_keywords': [],         # Short keywords with strict word boundaries
        'dotted_abbreviations': [],   # Keywords with dots (like abbreviations)
        'long_keywords_spaced': [],   # Multi-word phrases with spaces
        'flexible_phrases': [],       # Multi-word phrases with optional spaces/hyphens
        'flexible_mapping': {}        # Map flexible patterns back to original keywords
    }
    
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        
        if '.' in keyword_lower and len(keyword_lower) <= 4:
            # Dotted abbreviations (rare in our list)
            patterns['dotted_abbreviations'].append(keyword_lower)
        elif len(keyword_lower) <= 5 and ' ' not in keyword_lower:
            # Short keywords - need strict word boundaries
            patterns['short_keywords'].append(keyword_lower)
        elif ' ' in keyword_lower or '-' in keyword_lower:
            # Multi-word phrases - add both exact and flexible versions
            patterns['long_keywords_spaced'].append(keyword_lower)
            
            # Create flexible regex pattern: "artificial intelligence" -> "artificial[\\s_/.\\-]*intelligence" 
            # Include underscore, forward slash, and dot to match variants like "artificial_intelligence", "artificial/intelligence", "artificial.intelligence"
            flexible_pattern = re.sub(r'[\s\-]+', '[\\\\s_/\\.\\\\-]*', keyword_lower)
            patterns['flexible_phrases'].append(flexible_pattern)
            # Map the flexible pattern back to original keyword
            patterns['flexible_mapping'][flexible_pattern] = keyword_lower
        else:
            # Single long words
            patterns['long_keywords_spaced'].append(keyword_lower)
    
    # Remove duplicates (but preserve mapping)
    patterns['short_keywords'] = list(set(patterns['short_keywords']))
    patterns['dotted_abbreviations'] = list(set(patterns['dotted_abbreviations']))
    patterns['long_keywords_spaced'] = list(set(patterns['long_keywords_spaced']))
    
    print(f"SQL patterns prepared (matching stage_1 logic with flexible phrases):")
    print(f"  - Short keywords: {len(patterns['short_keywords'])}")
    print(f"  - Dotted abbreviations: {len(patterns['dotted_abbreviations'])}")
    print(f"  - Long spaced keywords: {len(patterns['long_keywords_spaced'])}")
    print(f"  - Flexible phrases: {len(patterns['flexible_phrases'])}")
    
    return patterns

def _execute_keyword_search(conn, patterns: dict, batch_size: int, limit_rows=None, test_companies_only=False) -> pd.DataFrame:
    """
    Execute the keyword search query matching stage_1 logic exactly.
    Uses proper word boundaries and excludes false positives.
    """
    
    # Build the WHERE clause matching stage_1 categorization
    where_conditions = []
    query_params = []
    
    # SHORT KEYWORDS: Strict word boundaries 
    if patterns['short_keywords']:
        short_patterns = []
        for keyword in patterns['short_keywords']:
            # 1) Normal contiguous match with strict word boundaries
            short_patterns.append("content_norm ~* %s")
            query_params.append(f'\\y{keyword}\\y')

            # 2) ALSO accept punct/dash between letters for very short tokens  
            if len(keyword) <= 5 and '.' not in keyword:
                punct_pat = _punct_between_letters_regex(keyword)
                short_patterns.append("content_clean ~* %s")
                query_params.append(punct_pat)

        if short_patterns:
            where_conditions.append(f"({' OR '.join(short_patterns)})")
    
    # DOTTED ABBREVIATIONS: Word boundaries for dotted terms  
    if patterns['dotted_abbreviations']:
        dotted_patterns = []
        for keyword in patterns['dotted_abbreviations']:
            dotted_patterns.append("content_norm ~* %s")
            # Escape dots for regex
            escaped_keyword = keyword.replace('.', '\\.')
            query_params.append(f'\\\\y{escaped_keyword}\\\\y')
        where_conditions.append(f"({' OR '.join(dotted_patterns)})")
    
    # LONG KEYWORDS / PHRASES
    if patterns['long_keywords_spaced']:
        spaced_patterns = []
        for keyword in patterns['long_keywords_spaced']:
            if ' ' in keyword or '-' in keyword:
                # multi-word phrase → substring is fine (you also have the flexible pattern)
                spaced_patterns.append("content_norm ILIKE %s")
                query_params.append(f'%{keyword}%')
            else:
                # single word → true word boundaries (Postgres \y)
                spaced_patterns.append("content_norm ~* %s")
                query_params.append(rf'\y{keyword}\y')
        where_conditions.append(f"({' OR '.join(spaced_patterns)})")
    
    # FLEXIBLE PHRASES: Multi-word phrases with optional spaces/hyphens
    if patterns['flexible_phrases']:
        flexible_patterns = []
        for pattern in patterns['flexible_phrases']:
            flexible_patterns.append("content_norm ~* %s")
            query_params.append(pattern)
        where_conditions.append(f"({' OR '.join(flexible_patterns)})")
    
    if not where_conditions:
        print("❌ No valid patterns generated")
        return pd.DataFrame()
    
    # Combine all conditions with OR (stage_1 matches if ANY pattern matches)
    full_where = ' OR '.join(where_conditions)
    
    # If testing on companies only, add company constraint
    additional_constraints = ""
    if test_companies_only:
        print("🧪 TEST MODE: Limiting to training companies for verification")
        additional_constraints = " AND company_name = ANY(%s)"
        query_params.append(TRAINING_COMPANIES)
    
    # Build the complete query
    query = f"""
    SELECT *
    FROM job_postings_unified 
    WHERE ({full_where}){additional_constraints}
    ORDER BY tst_created DESC;
    """
    
    print(f"Executing keyword search query...")
    print(f"Query length: {len(query)} characters")
    print(f"Parameters: {len(query_params)} values")
    
    try:
        # Execute query with parameters
        cur = conn.cursor()
        cur.execute(query, query_params)
        
        # Get column names
        colnames = [desc[0] for desc in cur.description]
        
        # Fetch all results (PostgreSQL handles memory management well)
        print("Fetching results from database...")
        rows = cur.fetchall()
        
        print(f"✅ Found {len(rows):,} matching job ads")
        
        # Create DataFrame
        print("Creating pandas DataFrame...")
        df = pd.DataFrame(rows, columns=colnames)
        
        # Add a column to track which keywords matched for each job
        print("Identifying matching keywords for each job (this may take a while)...")
        df['matched_keywords'] = df['content_clean'].apply(lambda x: _find_matching_keywords(x, patterns))
        print("✅ Keyword matching analysis complete")
        
        # Filter out jobs where no keywords were actually found (SQL false positives)
        initial_count = len(df)
        df_filtered = df[df['matched_keywords'].notna() & (df['matched_keywords'] != '')]
        filtered_count = len(df_filtered)
        removed_count = initial_count - filtered_count
        
        print(f"🔍 Filtered results:")
        print(f"  - Initial jobs from SQL: {initial_count:,}")
        print(f"  - Jobs with valid keywords: {filtered_count:,}")
        print(f"  - SQL false positives removed: {removed_count:,} ({removed_count/initial_count*100:.1f}%)")
        
        cur.close()
        return df_filtered
        
    except Exception as e:
        print(f"❌ Query execution failed: {str(e)}")
        if 'cur' in locals():
            cur.close()
        return pd.DataFrame()

def _find_matching_keywords(content: str, patterns: dict) -> str:
    """
    Find which specific keywords matched in the job content.
    Returns a comma-separated string of matched keywords.
    Enhanced with dotted equivalent support and improved error handling.
    """
    if not content or content != content:  # Handle NaN values
        return ""
    
    # Handle empty patterns gracefully
    if not patterns:
        return ""
    
    content_lower = str(content).lower()
    normalized_content = normalize_text_for_matching(content)
    matched = []
    
    # Check short keywords with word boundaries (accent and punctuation insensitive)
    for keyword in patterns.get('short_keywords', []):
        if (re.search(rf'\b{re.escape(keyword)}\b', content_lower) or
            re.search(rf'\b{re.escape(keyword)}\b', normalized_content)):
            matched.append(keyword)
        elif len(keyword) <= 5 and '.' not in keyword:
            punct_pat = _punct_between_letters_regex(keyword)
            if re.search(punct_pat, content_lower, flags=re.IGNORECASE):
                matched.append(f"{keyword}(punct)")
    
    # Check dotted abbreviations (accent and punctuation insensitive)
    for keyword in patterns.get('dotted_abbreviations', []):
        escaped_keyword = re.escape(keyword)
        # Use negative lookbehind/lookahead since word boundaries don't work well with dots
        if (re.search(rf'(?<!\w){escaped_keyword}(?!\w)', content_lower) or
            re.search(rf'\b{re.escape(keyword.replace(".", ""))}\b', normalized_content)):
            matched.append(keyword)
    
    # Check long keywords / phrases 
    for keyword in patterns.get('long_keywords_spaced', []):
        if ' ' in keyword or '-' in keyword:
            # multi-word phrase: substring is okay
            if keyword in content_lower or keyword in normalized_content:
                matched.append(keyword)
        else:
            # single word: boundaries that exclude underscores
            boundary = rf'(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])'
            if (re.search(boundary, content_lower) or
                re.search(boundary, normalized_content)):
                matched.append(keyword)
    
    # Check flexible phrases (with optional spaces/hyphens)
    for pattern in patterns.get('flexible_phrases', []):
        if (re.search(pattern, content_lower) or 
            re.search(pattern, normalized_content)):
            # Use proper mapping to get back to original keyword
            original_keyword = patterns.get('flexible_mapping', {}).get(pattern)
            if original_keyword and original_keyword not in matched:  # Avoid duplicates
                matched.append(f"{original_keyword}(flexible)")
    
    return ", ".join(sorted(set(matched)))

def extract_keyword_matching_jobs_batched(output_path="Data/", batch_months=6, batch_days=None):
    """
    Extract AI keyword matching jobs using date-based batching for better performance.
    Processes data in chunks by date ranges to avoid massive single queries.
    
    Parameters:
    -----------
    output_path : str, optional
        Directory path where to save the CSV file (default: "Data/")
    batch_months : int, optional
        Number of months to process in each batch (default: 6)
    batch_days : int, optional
        If specified, use days instead of months for smaller batches
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all job ads that match AI keywords
    """
    
    print("="*60)
    print("BATCHED AI KEYWORD EXTRACTION - OPTIMIZED APPROACH")
    print("="*60)
    
    # Load environment and connect
    load_dotenv('config.env')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    
    conn = psycopg2.connect(
        dbname=db_name, user=db_user, password=db_password,
        host=db_host, port=db_port
    )
    
    try:
        cur = conn.cursor()
        
        # Get date range of data
        cur.execute("SELECT MIN(tst_created), MAX(tst_created) FROM job_postings_unified WHERE tst_created IS NOT NULL;")
        min_date, max_date = cur.fetchone()
        print(f"Data range: {min_date} to {max_date}")
        
        # Create date batches
        from datetime import datetime, timedelta
        import calendar
        
        current_date = min_date
        batches = []
        
        while current_date < max_date:
            if batch_days:
                # Use day-based batching for testing
                end_date = current_date + timedelta(days=batch_days)
            else:
                # Add months to current date
                if current_date.month + batch_months <= 12:
                    end_date = current_date.replace(month=current_date.month + batch_months)
                else:
                    years_to_add = (current_date.month + batch_months - 1) // 12
                    new_month = (current_date.month + batch_months - 1) % 12 + 1
                    end_date = current_date.replace(year=current_date.year + years_to_add, month=new_month)
            
            if end_date > max_date:
                end_date = max_date
                
            batches.append((current_date, end_date))
            current_date = end_date
        
        if batch_days:
            print(f"Processing {len(batches)} date batches of ~{batch_days} days each")
        else:
            print(f"Processing {len(batches)} date batches of ~{batch_months} months each")
        
        # Process each batch with detailed progress monitoring
        import time
        all_results = []
        patterns = _prepare_sql_patterns(_get_ai_keywords())
        
        # Progress tracking
        start_time = time.time()
        total_found = 0
        
        for i, (start_date, end_date) in enumerate(batches):
            batch_start_time = time.time()
            
            if batch_days:
                date_format = '%Y-%m-%d'
            else:
                date_format = '%Y-%m'
            
            # Progress calculation
            progress_pct = (i / len(batches)) * 100
            
            print(f"\\n{'='*60}")
            print(f"🔄 BATCH {i+1}/{len(batches)} ({progress_pct:.1f}% complete)")
            print(f"📅 Date range: {start_date.strftime(date_format)} to {end_date.strftime(date_format)}")
            
            # Timing estimates
            if i > 0:
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / i
                remaining_batches = len(batches) - i
                estimated_remaining = avg_time_per_batch * remaining_batches
                
                print(f"⏱️  Avg time per batch: {avg_time_per_batch:.1f}s")
                print(f"⏰ Est. remaining time: {estimated_remaining/60:.1f} minutes")
                print(f"📊 Total found so far: {total_found:,} jobs")
            
            print(f"🔍 Searching batch...")
            
            batch_df = _execute_batched_keyword_search(conn, patterns, start_date, end_date)
            
            batch_time = time.time() - batch_start_time
            
            if len(batch_df) > 0:
                all_results.append(batch_df)
                total_found += len(batch_df)
                print(f"✅ Found {len(batch_df):,} AI jobs in this batch (took {batch_time:.1f}s)")
            else:
                print(f"❌ No AI jobs found in this batch (took {batch_time:.1f}s)")
        
        # Combine all results
        if all_results:
            print(f"\\n--- COMBINING RESULTS ---")
            final_df = pd.concat(all_results, ignore_index=True)
            
            # Remove duplicates (in case of date overlaps)
            initial_count = len(final_df)
            final_df = final_df.drop_duplicates(subset=['uid'])
            final_count = len(final_df)
            
            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} duplicates")
            
            # Save results
            filename = f'{output_path}ai_development_deduplicated.csv'
            final_df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f"\\n{'='*60}")
            print("BATCHED EXTRACTION COMPLETE")
            print(f"{'='*60}")
            print(f"✅ Total AI jobs: {len(final_df):,}")
            print(f"✅ Unique companies: {final_df['company_name'].nunique():,}")
            print(f"✅ Results saved to: {filename}")
            
            return final_df
        else:
            print("❌ No AI jobs found in any batch")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Batched extraction failed: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def _get_ai_keywords():
    """Return the AI keywords list for batched processing."""
    return load_multilingual_keywords()

def _execute_batched_keyword_search(conn, patterns: dict, start_date, end_date) -> pd.DataFrame:
    """
    Execute keyword search for a specific date range batch.
    Uses the same logic as the regular extraction by calling _execute_keyword_search
    with date filtering applied via a temporary view.
    """
    
    print(f"  🔍 Searching batch: {start_date} to {end_date}")
    
    try:
        cur = conn.cursor()
        
        # Create a temporary view with date filtering
        import time
        temp_view_name = f"batch_view_{int(time.time())}"
        cur.execute(f"""
            CREATE TEMP VIEW {temp_view_name} AS 
            SELECT * FROM job_postings_unified 
            WHERE tst_created >= %s AND tst_created < %s
        """, [start_date, end_date])
        
        # Temporarily modify the connection to use the filtered view
        # We'll execute the same logic but against the filtered data
        
        # Build the same WHERE clause as the regular extraction
        where_conditions = []
        query_params = []
        
        # SHORT KEYWORDS: Strict word boundaries 
        if patterns['short_keywords']:
            short_patterns = []
            for keyword in patterns['short_keywords']:
                # 1) Normal contiguous match with strict word boundaries
                short_patterns.append("content_norm ~* %s")
                query_params.append(f'\\y{keyword}\\y')

                # 2) ALSO accept punct/dash between letters for very short tokens
                if len(keyword) <= 5 and '.' not in keyword:
                    punct_pat = _punct_between_letters_regex(keyword)
                    short_patterns.append("content_clean ~* %s")
                    query_params.append(punct_pat)

            if short_patterns:
                where_conditions.append(f"({' OR '.join(short_patterns)})")
        
        # DOTTED ABBREVIATIONS: Word boundaries for dotted terms (exact match to regular extraction)
        if patterns['dotted_abbreviations']:
            dotted_patterns = []
            for keyword in patterns['dotted_abbreviations']:
                dotted_patterns.append("content_norm ~* %s")
                # Escape dots for regex
                escaped_keyword = keyword.replace('.', '\\.')
                query_params.append(f'\\y{escaped_keyword}\\y')
            where_conditions.append(f"({' OR '.join(dotted_patterns)})")
        
        # LONG KEYWORDS / PHRASES (exact match to regular extraction)
        if patterns['long_keywords_spaced']:
            spaced_patterns = []
            for keyword in patterns['long_keywords_spaced']:
                if ' ' in keyword or '-' in keyword:
                    # multi-word phrase → substring is fine (you also have the flexible pattern)
                    spaced_patterns.append("content_norm ILIKE %s")
                    query_params.append(f'%{keyword}%')
                else:
                    # single word → true word boundaries (Postgres \y)
                    spaced_patterns.append("content_norm ~* %s")
                    query_params.append(rf'\y{keyword}\y')
            where_conditions.append(f"({' OR '.join(spaced_patterns)})")
        
        # FLEXIBLE PHRASES: Multi-word phrases with optional spaces/hyphens (exact match to regular extraction)
        if patterns['flexible_phrases']:
            flexible_patterns = []
            for pattern in patterns['flexible_phrases']:
                flexible_patterns.append("content_norm ~* %s")
                query_params.append(pattern)
            where_conditions.append(f"({' OR '.join(flexible_patterns)})")
        
        if not where_conditions:
            cur.execute(f"DROP VIEW {temp_view_name}")
            return pd.DataFrame()
        
        # Build query using the temporary filtered view
        full_where = ' OR '.join(where_conditions)
        query = f"""
        SELECT *
        FROM {temp_view_name}
        WHERE ({full_where})
        ORDER BY tst_created DESC;
        """
        
        cur.execute(query, query_params)
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        # Clean up temporary view
        cur.execute(f"DROP VIEW {temp_view_name}")
        
        if rows:
            df = pd.DataFrame(rows, columns=colnames)
            # Add matched keywords column using the same function as regular extraction
            df['matched_keywords'] = df['content_clean'].apply(lambda x: _find_matching_keywords(x, patterns))

            # Filter out jobs where no keywords were actually found (SQL false positives)
            # This matches the filtering logic in _execute_keyword_search()
            initial_count = len(df)
            df_filtered = df[df['matched_keywords'].notna() & (df['matched_keywords'] != '')]
            filtered_count = len(df_filtered)
            removed_count = initial_count - filtered_count

            if removed_count > 0:
                print(f"🔍 Filtered batch results:")
                print(f"  - Initial jobs from SQL: {initial_count:,}")
                print(f"  - Jobs with valid keywords: {filtered_count:,}")
                print(f"  - SQL false positives removed: {removed_count:,} ({removed_count/initial_count*100:.1f}%)")

            return df_filtered
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Batch query failed: {str(e)}")
        # Try to clean up temp view if it exists
        try:
            if 'cur' in locals() and 'temp_view_name' in locals():
                cur.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
        except:
            pass
        return pd.DataFrame()
    finally:
        if 'cur' in locals():
            cur.close()

def extract_all_jobs_from_ai_companies_streaming(deduped_ai_jobs_file):
    """
    Extract ALL jobs from companies that appear in the deduplicated AI jobs dataset.
    Uses streaming approach: processes one company at a time and writes directly to CSV.
    This avoids memory issues while getting complete data (no job limits).
    
    Args:
        deduped_ai_jobs_file: Path to the deduplicated AI jobs CSV file
    
    Returns:
        Path to the output CSV file containing all jobs from AI companies
    """
    print("="*60)
    print("EXTRACTING ALL JOBS FROM AI COMPANIES (STREAMING)")
    print("="*60)
    
    # Load the deduplicated AI jobs file to get unique companies
    print(f"Loading AI companies from: {deduped_ai_jobs_file}")
    df_ai_jobs = pd.read_csv(deduped_ai_jobs_file)
    print(f"Loaded {len(df_ai_jobs):,} AI jobs")
    
    # Get unique company names
    unique_companies = df_ai_jobs['company_name'].dropna().unique().tolist()
    print(f"Found {len(unique_companies):,} unique AI companies")
    
    # Show sample companies
    print(f"\nSample companies:")
    for i, company in enumerate(unique_companies[:10]):
        print(f"  {i+1:2d}. {company}")
    if len(unique_companies) > 10:
        print(f"  ... and {len(unique_companies)-10:,} more companies")
    
    # Connect to database
    print(f"\n🔗 Connecting to database...")
    load_dotenv('config.env')
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None
    
    try:
        # Setup output file
        output_file = f"Data/all_jobs_from_ai_companies_streaming.csv"
        
        print(f"\n📝 Output file: {output_file}")
        print(f"🌊 Starting streaming processing: one company at a time...")
        
        # Track statistics
        total_jobs = 0
        total_companies_processed = 0
        companies_with_jobs = 0
        header_written = False
        
        # Process each company individually
        for company_idx, company_name in enumerate(unique_companies, 1):
            print(f"\n--- COMPANY {company_idx:,}/{len(unique_companies):,}: {company_name} ---")
            
            # Query ALL jobs for this single company (no limits!)
            query = """
            SELECT uid, title, company_name, content_clean, duplicate_group
            FROM job_postings_unified 
            WHERE company_name = %s
            ORDER BY tst_created DESC;
            """
            
            cur = conn.cursor()
            cur.execute(query, [company_name])
            
            # Fetch results for this company
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            
            if rows:
                # Create DataFrame for this company
                company_df = pd.DataFrame(rows, columns=colnames)
                
                # Deduplicate within this company
                initial_count = len(company_df)
                company_df = company_df.drop_duplicates(
                    subset=['company_name', 'title', 'content_clean'], 
                    keep='first'
                )
                deduped_count = len(company_df)
                duplicates_removed = initial_count - deduped_count
                
                print(f"  📊 {initial_count:,} total jobs → {deduped_count:,} unique jobs ({duplicates_removed:,} duplicates removed)")
                
                # Write to CSV (append mode after first company)
                mode = 'w' if not header_written else 'a'
                write_header = not header_written
                
                company_df.to_csv(output_file, mode=mode, header=write_header, index=False, encoding='utf-8')
                
                if not header_written:
                    header_written = True
                
                # Update statistics
                total_jobs += deduped_count
                companies_with_jobs += 1
                
                print(f"  ✅ Written {deduped_count:,} jobs to CSV")
            else:
                print(f"  ⚠️  No jobs found for {company_name}")
            
            total_companies_processed += 1
            
            # Progress update every 50 companies
            if company_idx % 50 == 0:
                print(f"\n🚀 PROGRESS: {company_idx:,}/{len(unique_companies):,} companies processed")
                print(f"   📈 Total jobs so far: {total_jobs:,}")
                print(f"   🏢 Companies with jobs: {companies_with_jobs:,}")
        
        # Final statistics
        print(f"\n{'='*60}")
        print("STREAMING EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Companies processed: {total_companies_processed:,}")
        print(f"✅ Companies with jobs: {companies_with_jobs:,}")
        print(f"✅ Total unique jobs extracted: {total_jobs:,}")
        print(f"✅ Average jobs per company: {total_jobs/companies_with_jobs:.1f}" if companies_with_jobs > 0 else "✅ No jobs found")
        print(f"📄 Output file: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Streaming extraction failed: {e}")
        return None
    finally:
        conn.close()
        print("🔐 Database connection closed")

def extract_all_jobs_from_ai_companies_batched(deduped_ai_jobs_file, batch_size=20):
    """
    DEPRECATED: Use extract_all_jobs_from_ai_companies_streaming instead.
    This version has a 500 job limit per company and can cause memory issues.
    """
    print("⚠️  WARNING: This function is deprecated and truncates data to 500 jobs per company.")
    print("⚠️  Use extract_all_jobs_from_ai_companies_streaming() for complete data extraction.")
    return extract_all_jobs_from_ai_companies_streaming(deduped_ai_jobs_file)

def extract_all_jobs_from_ai_companies(deduped_ai_jobs_file):
    """
    Extract ALL jobs from companies that appear in the deduplicated AI jobs dataset.
    Returns a CSV with all jobs from these companies (not just AI jobs).
    
    This is the original version - use extract_all_jobs_from_ai_companies_batched for large datasets.
    
    Args:
        deduped_ai_jobs_file: Path to the deduplicated AI jobs CSV file
    
    Returns:
        Path to the output CSV file containing all jobs from AI companies
    """
    return extract_all_jobs_from_ai_companies_batched(deduped_ai_jobs_file, batch_size=100)

def compare_and_extract_new_jobs(new_results_file, old_results_file):
    """
    Compare new stage_0 results with old stage_1 results and extract only NEW jobs.
    """
    print(f"Loading new results from {new_results_file}...")
    df_new = pd.read_csv(new_results_file)
    print(f"Loaded {len(df_new)} jobs from new results")
    
    print(f"Loading old results from {old_results_file}...")
    df_old = pd.read_csv(old_results_file)
    print(f"Loaded {len(df_old)} jobs from old results")
    
    # Compare using uid column
    old_uids = set(df_old['uid'].astype(str))
    new_uids = set(df_new['uid'].astype(str))
    
    print(f"\nComparison:")
    print(f"  - Old results: {len(old_uids)} unique UIDs")
    print(f"  - New results: {len(new_uids)} unique UIDs")
    
    # Find UIDs that are in new but not in old
    new_only_uids = new_uids - old_uids
    print(f"  - Jobs only in NEW results: {len(new_only_uids)}")
    
    # Filter to get only the new jobs
    df_new_only = df_new[df_new['uid'].astype(str).isin(new_only_uids)].copy()
    
    # Select columns for manual review  
    review_columns = ['uid', 'title', 'company_name', 'matched_keywords', 'content_clean']
    if 'matched_keywords' in df_new_only.columns:
        df_review = df_new_only[review_columns].copy()
    else:
        print("❌ No matched_keywords column found, using available columns")
        df_review = df_new_only.copy()
    
    # Sort by company and title
    df_review = df_review.sort_values(['company_name', 'title'])
    
    # Show analysis
    print(f"\nNEW jobs by company:")
    print(df_new_only['company_name'].value_counts().head(10))
    
    if 'matched_keywords' in df_new_only.columns:
        print(f"\nMost common keywords in NEW jobs:")
        all_keywords = []
        for keywords_str in df_new_only['matched_keywords'].dropna():
            if keywords_str:
                keywords = [k.strip() for k in keywords_str.split(',')]
                all_keywords.extend(keywords)
        
        if all_keywords:
            keyword_counts = pd.Series(all_keywords).value_counts()
            print(keyword_counts.head(20))
    
    return df_review

# Example usage if run directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-test-set":
        # Create test set of 50 companies
        df_test = create_test_set(num_companies=50)
        print(f"Created test set with {len(df_test)} job ads from 50 companies (excluding original 100 training companies)")
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract-keywords":
        # Extract all AI keyword matching jobs from full database
        print("🚀 Starting full database AI keyword extraction...")
        df_ai_jobs = extract_keyword_matching_jobs()
        if len(df_ai_jobs) > 0:
            print(f"✅ Successfully extracted {len(df_ai_jobs):,} AI-related job ads")
        else:
            print("❌ No AI-related jobs found")
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract-keywords-batched":
        # Extract with batched approach for better performance
        print("🚀 Starting BATCHED database AI keyword extraction...")
        df_ai_jobs = extract_keyword_matching_jobs_batched()
        if len(df_ai_jobs) > 0:
            print(f"✅ Successfully extracted {len(df_ai_jobs):,} AI-related job ads")
        else:
            print("❌ No AI-related jobs found")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-single-batch":
        # Test single batch execution with known good date range
        print("🧪 Testing SINGLE BATCH execution...")
        from datetime import datetime
        import pandas as pd
        
        # Test with first day that we know has AI jobs
        start_date = '2012-01-01'
        end_date = '2012-01-02'
        
        load_dotenv('config.env')
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        
        print(f"Testing batch: {start_date} to {end_date}")
        
        # Get keyword patterns (proper structure)
        patterns = {
            'short_keywords': ['ai', 'ml'],
            'dotted_abbreviations': [],
            'long_keywords_spaced': [],
            'flexible_phrases': [],
            'flexible_mapping': {}
        }
        
        batch_df = _execute_batched_keyword_search(conn, patterns, start_date, end_date)
        print(f"✅ Single batch test result: {len(batch_df)} AI jobs found")
        
        if len(batch_df) > 0:
            print("Sample jobs:")
            print(batch_df[['uid', 'company_name', 'tst_created']].head(3))
        
        conn.close()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-simple-batches":
        # Test with simplified keyword list to verify performance
        print("🧪 Testing SIMPLE BATCHES with reduced keywords...")
        
        # Temporarily modify the keyword list for testing
        original_get_ai_keywords = _get_ai_keywords
        def _get_ai_keywords_simple():
            return ['ai', 'ml', 'pytorch', 'tensorflow', 'machine learning']
        
        # Monkey patch for testing
        import types
        globals()['_get_ai_keywords'] = _get_ai_keywords_simple
        
        df_ai_jobs = extract_keyword_matching_jobs_batched(batch_days=7)  # 7 days with simple keywords
        
        # Restore original function
        globals()['_get_ai_keywords'] = original_get_ai_keywords
        
        if len(df_ai_jobs) > 0:
            print(f"✅ Simple batches test successful: {len(df_ai_jobs):,} AI-related job ads")
        else:
            print("❌ Simple batches test found no AI jobs")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-micro-batches":
        # Test with very small time windows using full keyword set
        print("🧪 Testing MICRO BATCHES (6 hours each) with FULL keywords...")
        df_ai_jobs = extract_keyword_matching_jobs_batched(batch_days=0.25)  # 6 hours
        if len(df_ai_jobs) > 0:
            print(f"✅ Micro batches test successful: {len(df_ai_jobs):,} AI-related job ads")
        else:
            print("❌ Micro batches test found no AI jobs")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-tiny-batches":
        # Test with very small day-based batches to verify approach works  
        print("🧪 Testing TINY BATCHES (1 day each) to verify batched approach...")
        df_ai_jobs = extract_keyword_matching_jobs_batched(batch_days=1)
        if len(df_ai_jobs) > 0:
            print(f"✅ Tiny batches test successful: {len(df_ai_jobs):,} AI-related job ads")
        else:
            print("❌ Tiny batches test found no AI jobs")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-keywords":
        # Test keyword matching on training companies only
        print("🧪 Starting test keyword extraction on training companies only...")
        df_ai_jobs = extract_keyword_matching_jobs(test_companies_only=True)
        if len(df_ai_jobs) > 0:
            print(f"✅ Successfully extracted {len(df_ai_jobs):,} AI-related job ads from training companies")
        else:
            print("❌ No AI-related jobs found in training companies")
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract-ai-company-jobs":
        # Extract ALL jobs from companies that appear in deduplicated AI jobs dataset
        if len(sys.argv) < 3:
            print("Usage: python3 stage_0_get_job_ads.py --extract-ai-company-jobs <deduped_ai_jobs_file>")
            print("Example: python3 stage_0_get_job_ads.py --extract-ai-company-jobs Data/ai_development_deduplicated_custom_20250829_164626.csv")
        else:
            deduped_file = sys.argv[2]
            print(f"🏢 Extracting all jobs from AI companies in: {deduped_file}")
            output_file = extract_all_jobs_from_ai_companies(deduped_file)
            if output_file:
                print(f"✅ Extraction complete. Results saved to: {output_file}")
            else:
                print("❌ Extraction failed")
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Compare new and old results to extract only NEW jobs
        if len(sys.argv) < 4:
            print("Usage: python3 stage_0_get_job_ads.py --compare <new_file> <old_file>")
            print("Example: python3 stage_0_get_job_ads.py --compare Data/full_job_ads_20250808_123424.csv Data/enhanced_ai_detection_test_20250722_160703.csv")
        else:
            new_file = sys.argv[2]
            old_file = sys.argv[3]
            print(f"🔍 Comparing {new_file} vs {old_file} to extract NEW jobs...")
            df_new_jobs = compare_and_extract_new_jobs(new_file, old_file)
            print(f"✅ Comparison complete. Found {len(df_new_jobs)} new jobs.")
            
            # Save the new jobs for manual review
            output_file = f"Data/new_jobs_only.csv" 
            df_new_jobs.to_csv(output_file, index=False)
            print(f"💾 NEW jobs saved to {output_file} for manual inspection")
    else:
        # Original training set creation (unchanged)
        df_random_ads = load_company_ads(num_companies=100, output_path="random_company_ads")
        print(f"Retrieved {len(df_random_ads)} job ads from 100 random companies")