#!/usr/bin/env python3
"""
Batch Audio Processing CSV Logger
Processes local audio files through batch endpoint and saves exact API response to CSV
"""

import requests
import csv
import os
import glob
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Configuration
API_BASE = "http://localhost:8030"
# TODO: Update this to your actual audio files directory
AUDIO_FILES_DIR = r"C:\Users\Plyra\Downloads\high_lat"
# TODO: Update this to your actual input CSV file path
INPUT_CSV = "test_input.csv"
OUTPUT_DIR = "csv_outputs"
CSV_FILENAME = f"audio_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# CSV column headers - exact match to MetricsResponse schema
CSV_HEADERS = [
    'file_path',
    'filename',
    'status',
    'error_message',
    'latency_points',
    'average_latency',
    'p50_latency',
    'p90_latency',
    'min_latency',
    'max_latency',
    'ai_interrupting_user',
    'user_interrupting_ai',
    'ai_user_overlap_count',
    'user_ai_overlap_count',
    'talk_ratio',
    'average_pitch',
    'words_per_minute',
    'personaAdherence',
    'languageSwitch',
    'sentiment'
]

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def get_audio_data_from_csv(csv_path: str, audio_dir: str) -> List[Dict[str, str]]:
    """Get audio file paths and agent_ids from the specified CSV file"""
    audio_data = []
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        return []
        
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            agent_id = row.get('agent_id')
            filename = row.get('filename')
            if agent_id and filename:
                file_path = os.path.join(audio_dir, filename)
                if os.path.exists(file_path):
                    audio_data.append({"path": file_path, "agent_id": agent_id})
                else:
                    print(f"⚠️  Audio file not found: {file_path}")
            else:
                print(f"⚠️  Skipping invalid row in CSV: {row}")

    print(f"Found {len(audio_data)} audio files with agent IDs from {csv_path}")
    for data in audio_data[:5]:
        print(f"  - {os.path.basename(data['path'])} (Agent: {data['agent_id']})")
    if len(audio_data) > 5:
        print(f"  ... and {len(audio_data) - 5} more files")
        
    return audio_data

def process_batch(audio_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Process files through the batch API endpoint"""
    payload = {"files": audio_data}
    
    try:
        print(f"📤 Processing {len(audio_data)} files through batch endpoint...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/batch",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Processing completed in {processing_time:.2f} seconds")
        print(f"📊 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            results = response_data.get('results', [])
            
            success_count = len([r for r in results if r.get('status') == 'success'])
            error_count = len(results) - success_count
            print(f"✅ {success_count} successful, {error_count} errors")
            
            return results
        else:
            print(f"❌ API Error: {response.text}")
            return []
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def save_to_csv(results: List[Dict[str, Any]], csv_path: str):
    """Save results to CSV - exact API response structure"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS, extrasaction='ignore')
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"✅ Results saved to: {csv_path}")

def print_summary(results: List[Dict[str, Any]]):
    """Print processing summary"""
    if not results:
        print("📊 No results to summarize")
        return
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    print(f"\n📊 PROCESSING SUMMARY")
    print(f"=" * 50)
    print(f"Total Files: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        avg_latency = sum(r.get('average_latency', 0) for r in successful) / len(successful)
        avg_ratio = sum(r.get('talk_ratio', 0) for r in successful) / len(successful)
        print(f"📈 Average Latency: {avg_latency:.2f}ms")
        print(f"🗣️  Average Talk Ratio: {avg_ratio:.2f}")
    
    if failed:
        print(f"\n❌ FAILED FILES:")
        for result in failed[:3]:
            filename = result.get('filename', 'Unknown')
            error = result.get('error_message', 'Unknown error')
            print(f"  - {filename}: {error}")
        if len(failed) > 3:
            print(f"  ... and {len(failed) - 3} more failures")

def main():
    """Main processing function"""
    print("🚀 Batch Audio Processing CSV Logger")
    print("=" * 50)
    
    # Setup
    ensure_output_directory()
    csv_output_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    
    # Get audio files
    print(f"\n📁 Scanning for {INPUT_CSV}...")
    audio_data = get_audio_data_from_csv(INPUT_CSV, AUDIO_FILES_DIR)
    
    if not audio_data:
        print("❌ No audio files found to process")
        return
    
    # Process files
    print(f"\n🔄 Processing {len(audio_data)} files...")
    results = process_batch(audio_data)
    
    if results:
        # Save to CSV
        save_to_csv(results, csv_output_path)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Processing complete!")
        print(f"📄 CSV file: {os.path.abspath(csv_output_path)}")
    else:
        print("❌ No results to save")

if __name__ == "__main__":
    main()
