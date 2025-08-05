#!/usr/bin/env python3
"""
Monitor the progress of the full database processing
"""

import time
import os
import subprocess
from pathlib import Path

def check_process_status():
    """Check if the full database process is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'run_full_database_balanced.py' in result.stdout:
            print("✅ Full database processing is running")
            return True
        else:
            print("❌ Full database processing is not running")
            return False
    except:
        return False

def check_output_files():
    """Check if any output files have been created"""
    data_dir = Path(__file__).parent.parent / "data"
    
    output_files = [
        "full_database_ai_jobs_balanced_cleaning.csv",
        "full_database_ai_jobs_summary.csv"
    ]
    
    found_files = []
    for filename in output_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            found_files.append(f"  📄 {filename} ({size_mb:.1f} MB)")
    
    if found_files:
        print("📁 Output files found:")
        for file_info in found_files:
            print(file_info)
    else:
        print("📁 No output files created yet")
    
    return len(found_files) > 0

def check_log_file():
    """Check the log file for any content"""
    log_file = Path("full_database_run.log")
    
    if log_file.exists():
        size = log_file.stat().st_size
        if size > 0:
            print(f"📝 Log file exists ({size} bytes)")
            # Try to read last few lines
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("📝 Latest log entries:")
                        for line in lines[-5:]:
                            print(f"    {line.strip()}")
                    else:
                        print("📝 Log file is empty")
            except:
                print("📝 Could not read log file")
        else:
            print("📝 Log file exists but is empty")
    else:
        print("📝 No log file found")

def estimate_completion():
    """Estimate completion time based on subset results"""
    # Based on subset results: 100k records processed in ~11 minutes
    total_records = 13785774
    subset_records = 100000
    subset_time_minutes = 11
    
    estimated_total_minutes = (total_records / subset_records) * subset_time_minutes
    estimated_hours = estimated_total_minutes / 60
    
    print(f"⏱️  Estimated total processing time: {estimated_total_minutes:.0f} minutes ({estimated_hours:.1f} hours)")
    print(f"📊 Based on: {total_records:,} total records, {subset_records:,} subset processed in {subset_time_minutes} minutes")

def main():
    """Monitor the full database processing"""
    print("="*60)
    print("FULL DATABASE PROCESSING MONITOR")
    print("="*60)
    
    # Check process status
    is_running = check_process_status()
    
    # Check for output files
    has_outputs = check_output_files()
    
    # Check log file
    check_log_file()
    
    # Show estimates
    estimate_completion()
    
    print("\n" + "="*60)
    if is_running:
        print("🚀 Processing is in progress...")
        print("💡 This will take approximately 25-30 minutes to complete")
        print("📈 Expected to find ~60,000 AI-related jobs")
        print("🔄 Check again in 10-15 minutes for updates")
    elif has_outputs:
        print("✅ Processing appears to be complete!")
        print("📁 Output files have been created")
    else:
        print("⚠️  Processing may have stopped or not started properly")
        print("🔧 You may need to restart the process")

if __name__ == "__main__":
    main()