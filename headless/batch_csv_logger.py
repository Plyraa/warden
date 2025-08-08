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
import argparse
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
    'hasEcho',
    'echoInterrupt',
    'hasNoise',
    'noiseInterrupt',
    'conversation_type',
    'initial_latency_points',
    'languageSwitch',
    'sentiment',
    'negativeExperience',
    'negativeExperienceReasoning',
    'userRepetition',
    'agentRepetition',
    'taskCompletion',
    'taskCompletionReasoning'
]

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def get_audio_data_from_csv(csv_path: str, audio_dir: str) -> List[Dict[str, str]]:
    """Get audio file paths from the specified CSV file"""
    audio_data = []
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        return []
        
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row.get('filename')
            if filename:
                file_path = os.path.join(audio_dir, filename)
                if os.path.exists(file_path):
                    item = {"path": file_path}
                    # Optional: pass conversation_type from CSV if present
                    conv = row.get('conversation_type')
                    if conv:
                        item["conversation_type"] = conv
                    audio_data.append(item)
                else:
                    print(f"⚠️  Audio file not found: {file_path}")
            else:
                print(f"⚠️  Skipping invalid row in CSV: {row}")

    print(f"Found {len(audio_data)} audio files from {csv_path}")
    for data in audio_data[:5]:
        print(f"  - {os.path.basename(data['path'])}")
    if len(audio_data) > 5:
        print(f"  ... and {len(audio_data) - 5} more files")
        
    return audio_data

def process_batch(audio_data: List[Dict[str, str]], run_behavioral: bool = False) -> List[Dict[str, Any]]:
    """Process files through the batch API endpoint"""
    payload = {"files": audio_data}
    
    try:
        print(f"📤 Processing {len(audio_data)} files through batch endpoint...")
        start_time = time.time()
        
        params = {"run_behavioral": str(run_behavioral).lower()} if run_behavioral else None
        response = requests.post(
            f"{API_BASE}/batch",
            params=params,
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

def print_results_table(results: List[Dict[str, Any]]):
    """Print results in a pretty table format using only built-in Python"""
    if not results:
        print("📊 No results to display")
        return
    
    print(f"\n📊 DETAILED RESULTS TABLE")
    print("=" * 140)
    
    # Define column headers and widths
    headers = ["Filename", "Status", "Latency", "Ratio", "Echo", "Noise", "Sentiment", "NegExp", "AgentRep", "UserRep", "TaskComp"]
    widths = [22, 8, 10, 8, 6, 6, 10, 8, 9, 8, 12]
    
    # Print header
    header_line = "│"
    for i, (header, width) in enumerate(zip(headers, widths)):
        header_line += f" {header:^{width}} │"
    print(header_line)
    
    # Print separator line
    separator = "├"
    for width in widths:
        separator += "─" * (width + 2) + "┼"
    separator = separator[:-1] + "┤"
    print(separator)
    
    # Print data rows
    for result in results:
        # Prepare data
        filename = result.get('filename', 'Unknown')[:23] + ".." if len(result.get('filename', '')) > 25 else result.get('filename', 'Unknown')
        status = "✅ OK" if result.get('status') == 'success' else "❌ ERR"
        
        if result.get('status') == 'success':
            latency = f"{result.get('average_latency', 0):.1f}ms"
            ratio = f"{result.get('talk_ratio', 0):.2f}"
        else:
            latency = "N/A"
            ratio = "N/A"
        
        # Handle boolean values with emojis
        echo = "🔊" if result.get('hasEcho') else "🔇"
        noise = "📢" if result.get('hasNoise') else "🔕"
        
        # Handle optional fields
        sentiment = result.get('sentiment', 'N/A')[:10] if result.get('sentiment') else 'N/A'
        
        negative_exp = result.get('negativeExperience')
        if negative_exp is True:
            neg_exp = "⚠️ YES"
        elif negative_exp is False:
            neg_exp = "✅ NO"
        else:
            neg_exp = "N/A"
        
        agent_rep = result.get('agentRepetition')
        if agent_rep is True:
            agent_rep_str = "🔄 YES"
        elif agent_rep is False:
            agent_rep_str = "✅ NO"
        else:
            agent_rep_str = "N/A"
        
        user_rep = result.get('userRepetition')
        if user_rep is True:
            user_rep_str = "🔄 YES"
        elif user_rep is False:
            user_rep_str = "✅ NO"
        else:
            user_rep_str = "N/A"
        
        # Handle task completion
        task_comp = result.get('taskCompletion')
        if task_comp == "Fully Completed":
            task_comp_str = "✅ FULL"
        elif task_comp == "Partially Completed":
            task_comp_str = "⚠️ PART"
        elif task_comp == "Not Completed":
            task_comp_str = "❌ NONE"
        else:
            task_comp_str = "N/A"
        
        # Prepare row data
        row_data = [filename, status, latency, ratio, echo, noise, sentiment, neg_exp, agent_rep_str, user_rep_str, task_comp_str]
        
        # Print row
        row_line = "│"
        for i, (data, width) in enumerate(zip(row_data, widths)):
            if i == 0:  # Left align filename
                row_line += f" {data:<{width}} │"
            else:  # Center align others
                row_line += f" {data:^{width}} │"
        print(row_line)
    
    # Print bottom border
    bottom = "└"
    for width in widths:
        bottom += "─" * (width + 2) + "┴"
    bottom = bottom[:-1] + "┘"
    print(bottom)
    
    # Add legend
    print(f"\n📖 LEGEND:")
    print(f"Status: ✅ Success, ❌ Error")
    print(f"Audio: 🔊 Echo, 🔇 No Echo, 📢 Noise, 🔕 No Noise")
    print(f"Risk: ⚠️ Risk Detected, ✅ No Risk, 🔄 Repetition")
    print(f"Task: ✅ FULL = Fully Completed, ⚠️ PART = Partially Completed, ❌ NONE = Not Completed, N/A = Not Analyzed")

def main():
    """Main processing function"""
    print("🚀 Batch Audio Processing CSV Logger")
    print("=" * 50)
    
    # Args
    parser = argparse.ArgumentParser(description="Batch audio metrics logger")
    parser.add_argument("--run-behavioral", action="store_true", help="Enable LLM-based behavioral metrics")
    args = parser.parse_args()

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

    # 1) Run baseline (audio metrics + scripted init)
    print("\n⚙️ Behavioral metrics: DISABLED")
    results = process_batch(audio_data, run_behavioral=False)
    if results:
        save_to_csv(results, csv_output_path)
        print_summary(results)
        print_results_table(results)
        print(f"\n✅ Baseline complete")
        print(f"📄 Metrics CSV: {os.path.abspath(csv_output_path)}")
    else:
        print("❌ No baseline results to save")

    # 2) Run behavioral-enabled pass for LLM metrics
    print("\n⚙️ Behavioral metrics: ENABLED")
    results_llm = process_batch(audio_data, run_behavioral=True)
    if results_llm:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        llm_csv_path = os.path.join(OUTPUT_DIR, f"llm_analysis_{ts}.csv")
        save_to_csv(results_llm, llm_csv_path)
        print_summary(results_llm)
        print_results_table(results_llm)
        print(f"\n✅ Behavioral pass complete")
        print(f"📄 LLM CSV: {os.path.abspath(llm_csv_path)}")
    else:
        print("❌ No behavioral results to save")

if __name__ == "__main__":
    main()
