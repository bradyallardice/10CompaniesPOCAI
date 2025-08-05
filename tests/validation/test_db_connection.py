#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

# Database connection details from environment
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')

print(f"Attempting to connect to database: {db_name} on {db_host}:{db_port}")

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    
    cur = conn.cursor()
    
    # Simple test query
    cur.execute("SELECT COUNT(*) FROM job_postings WHERE content_clean IS NOT NULL;")
    count = cur.fetchone()[0]
    
    print(f"Database connection successful!")
    print(f"Total job postings with content_clean: {count}")
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"Database connection failed: {e}")