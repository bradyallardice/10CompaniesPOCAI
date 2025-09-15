#!/usr/bin/env python3
"""
Update Panel Data - Load JSON files into PostgreSQL database

This script processes the downloaded panel data files (data_*.json.gzip) and loads them
into the job_postings table, handling duplicates and data type conversions.
"""

import gzip
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from pathlib import Path

class PanelDataLoader:
    """Load panel data from JSON files into PostgreSQL database"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv('config.env')
        
        # Database connection parameters
        self.db_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        print(f"Connecting to database: {self.db_params['database']} at {self.db_params['host']}")
        
    def connect_db(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.autocommit = False  # We want transaction control
            return conn
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise
            
    def parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[str]:
        """Convert timestamp string to PostgreSQL format"""
        if not timestamp_str:
            return None
        try:
            # Handle various timestamp formats
            if timestamp_str.endswith(' UTC'):
                timestamp_str = timestamp_str.replace(' UTC', '+00:00')
            return timestamp_str
        except Exception as e:
            print(f"Warning: Could not parse timestamp '{timestamp_str}': {e}")
            return None
    
    def process_record(self, record: Dict) -> Dict:
        """Process a single JSON record for database insertion"""
        processed = {}
        
        # Map duplicate_group to uid (primary key)
        processed['uid'] = record.get('duplicate_group')
        processed['title'] = record.get('title')
        processed['url'] = record.get('url')
        processed['content_clean'] = record.get('content_clean')
        processed['company_id'] = record.get('company_id')
        processed['company_name'] = record.get('company_name')
        processed['company_size'] = record.get('company_size')
        processed['company_is_recruiter'] = record.get('company_is_recruiter')
        
        # Handle timestamps
        processed['tst_created'] = self.parse_timestamp(record.get('tst_created'))
        processed['tst_deleted'] = self.parse_timestamp(record.get('tst_deleted'))
        
        # Handle JSON/JSONB fields - convert lists to JSON strings
        processed['x28_industries'] = json.dumps(record.get('x28_industries', []))
        processed['x28_occupations'] = json.dumps(record.get('x28_occupations', []))
        processed['cantons'] = json.dumps(record.get('cantons', []))
        
        processed['location_raw'] = record.get('location_raw')
        processed['duplicate_group'] = record.get('duplicate_group')
        
        return processed
    
    def load_json_file(self, filepath: str) -> List[Dict]:
        """Load and parse a gzipped JSON file"""
        print(f"Loading file: {filepath}")
        records = []
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # Skip empty lines
                            record = json.loads(line.strip())
                            processed_record = self.process_record(record)
                            records.append(processed_record)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return []
            
        print(f"Loaded {len(records)} records from {filepath}")
        return records
    
    def insert_records(self, conn, records: List[Dict]) -> tuple:
        """Insert records into database with conflict handling"""
        if not records:
            return 0, 0
            
        print(f"Inserting {len(records)} records...")
        
        # Prepare the SQL statement with UPSERT (ON CONFLICT DO UPDATE)
        columns = [
            'uid', 'title', 'url', 'content_clean', 'company_id', 'company_name',
            'company_size', 'company_is_recruiter', 'tst_created', 'tst_deleted',
            'x28_industries', 'x28_occupations', 'location_raw', 'cantons', 'duplicate_group'
        ]
        
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)
        
        # Create conflict resolution - update all fields except uid on conflict
        update_columns = [col for col in columns if col != 'uid']
        updates = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
        
        insert_sql = f"""
        INSERT INTO job_postings ({columns_str})
        VALUES %s
        ON CONFLICT (uid) DO UPDATE SET
        {updates}
        """
        
        # Prepare data tuples
        data_tuples = []
        for record in records:
            tuple_data = tuple(record.get(col) for col in columns)
            data_tuples.append(tuple_data)
        
        try:
            cursor = conn.cursor()
            
            # Get initial count
            cursor.execute("SELECT COUNT(*) FROM job_postings")
            initial_count = cursor.fetchone()[0]
            
            # Execute the batch insert
            execute_values(
                cursor, insert_sql, data_tuples,
                template=None, page_size=1000
            )
            
            # Get final count
            cursor.execute("SELECT COUNT(*) FROM job_postings")
            final_count = cursor.fetchone()[0]
            
            # Calculate statistics
            inserted_count = final_count - initial_count
            updated_count = len(records) - inserted_count
            
            conn.commit()
            cursor.close()
            
            print(f"Database operation completed:")
            print(f"  - Records inserted: {inserted_count}")
            print(f"  - Records updated: {updated_count}")
            print(f"  - Total records in database: {final_count}")
            
            return inserted_count, updated_count
            
        except Exception as e:
            conn.rollback()
            print(f"Database insert failed: {e}")
            raise
    
    def load_all_files(self):
        """Load all panel data files in the current directory"""
        print("=== Panel Data Loader ===")
        print(f"Started at: {datetime.now()}")
        
        # Find all data files
        data_files = sorted([f for f in os.listdir('.') if f.startswith('data_') and f.endswith('.json.gzip')])
        
        if not data_files:
            print("No data files found! Expected files like data_*.json.gzip")
            return
            
        print(f"Found {len(data_files)} data files: {data_files}")
        
        # Connect to database
        conn = self.connect_db()
        
        try:
            total_inserted = 0
            total_updated = 0
            
            for file_path in data_files:
                print(f"\n--- Processing {file_path} ---")
                
                # Load records from file
                records = self.load_json_file(file_path)
                
                if records:
                    # Insert into database
                    inserted, updated = self.insert_records(conn, records)
                    total_inserted += inserted
                    total_updated += updated
                else:
                    print(f"No valid records found in {file_path}")
            
            print(f"\n=== SUMMARY ===")
            print(f"Files processed: {len(data_files)}")
            print(f"Total records inserted: {total_inserted}")
            print(f"Total records updated: {total_updated}")
            print(f"Completed at: {datetime.now()}")
            
        except Exception as e:
            print(f"Error during processing: {e}")
            raise
        finally:
            conn.close()

def main():
    """Main function"""
    try:
        loader = PanelDataLoader()
        loader.load_all_files()
        print("\n✅ Panel data update completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Panel data update failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())