#!/usr/bin/env python3
"""
Fast LLM Logger
Processes local audio files for complete LLM evaluation including behavioral analysis
"""

import os
import csv
import glob
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from llm_evaluator import LlmEvaluator

# Configuration
AUDIO_FILES_DIR = r"C:\Users\ArdAlp\Downloads\high_lat"  # TODO: Update this to your audio files directory
INPUT_CSV = "test_input.csv"  # TODO: Update this to your input CSV file path
OUTPUT_DIR = "csv_outputs"
CSV_FILENAME = f"llm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# CSV column headers for complete LLM evaluation
CSV_HEADERS = [
    'filename',
    'file_path',
    'status',
    'error_message',
    # LLM Evaluation metrics
    'languageSwitch',
    'sentiment',
    # Behavioral Analysis metrics
    'userChurnRisk',
    'userChurnReasoning',
    'userRepetition',
    'agentRepetition',
    'taskCompletion',
    'taskCompletionReasoning',
    'processing_time_seconds'
]

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def get_audio_files_from_csv(csv_path: str, audio_dir: str) -> List[str]:
    """Get audio file paths from the specified CSV file"""
    audio_files = []
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        # Fallback to scanning directory for MP3 files
        pattern = os.path.join(audio_dir, "*.mp3")
        audio_files = glob.glob(pattern)
        print(f"📁 Scanning directory instead, found {len(audio_files)} MP3 files")
        return audio_files
        
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row.get('filename')
            if filename:
                file_path = os.path.join(audio_dir, filename)
                if os.path.exists(file_path):
                    audio_files.append(file_path)
                else:
                    print(f"⚠️  Audio file not found: {file_path}")
            else:
                print(f"⚠️  Skipping invalid row in CSV: {row}")

    print(f"Found {len(audio_files)} audio files from {csv_path}")
    for file_path in audio_files[:5]:
        print(f"  - {os.path.basename(file_path)}")
    if len(audio_files) > 5:
        print(f"  ... and {len(audio_files) - 5} more files")
        
    return audio_files

def process_file_llm_evaluation(evaluator: LlmEvaluator, file_path: str) -> Dict[str, Any]:
    """Process a single file for complete LLM evaluation"""
    start_time = time.time()
    filename = os.path.basename(file_path)
    
    try:
        print(f"🧠 Analyzing: {filename}")
        
        # Run combined LLM evaluation (basic + behavioral)
        result = evaluator.run_combined_evaluation(file_path)
        
        processing_time = time.time() - start_time
        
        if result:
            # Extract results directly from CombinedEvaluationResult
            return {
                'filename': filename,
                'file_path': file_path,
                'status': 'success',
                'error_message': None,
                # LLM Evaluation results
                'languageSwitch': result.languageSwitch,
                'sentiment': result.sentiment,
                # Behavioral Analysis results
                'userChurnRisk': result.userChurnRisk,
                'userChurnReasoning': result.userChurnReasoning,
                'userRepetition': result.userRepetition,
                'agentRepetition': result.agentRepetition,
                'taskCompletion': result.taskCompletion,
                'taskCompletionReasoning': result.taskCompletionReasoning,
                'processing_time_seconds': round(processing_time, 2)
            }
        else:
            return {
                'filename': filename,
                'file_path': file_path,
                'status': 'error',
                'error_message': 'Evaluation returned no results',
                'languageSwitch': None,
                'sentiment': None,
                'userChurnRisk': None,
                'userChurnReasoning': None,
                'userRepetition': None,
                'agentRepetition': None,
                'taskCompletion': None,
                'taskCompletionReasoning': None,
                'processing_time_seconds': round(processing_time, 2)
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        print(f"❌ {filename}: {error_msg}")
        
        return {
            'filename': filename,
            'file_path': file_path,
            'status': 'error',
            'error_message': error_msg,
            'languageSwitch': None,
            'sentiment': None,
            'userChurnRisk': None,
            'userChurnReasoning': None,
            'userRepetition': None,
            'agentRepetition': None,
            'taskCompletion': None,
            'taskCompletionReasoning': None,
            'processing_time_seconds': round(processing_time, 2)
        }

def process_batch_llm_evaluation(audio_files: List[str]) -> List[Dict[str, Any]]:
    """Process multiple files for complete LLM evaluation"""
    # Initialize the LLM evaluator
    try:
        evaluator = LlmEvaluator()
        print("✅ LLM evaluator initialized with unified Gemini semantic analysis")
    except Exception as e:
        print(f"❌ Failed to initialize LLM evaluator: {e}")
        return []
    
    results = []
    total_files = len(audio_files)
    
    print(f"\n🚀 Starting LLM evaluation for {total_files} files")
    start_time = time.time()
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"\n📊 Processing file {i}/{total_files}")
        
        result = process_file_llm_evaluation(evaluator, file_path)
        results.append(result)
        
        if result['status'] == 'success':
            # Show all results since we always do complete analysis
            print(f"✅ {result['filename']}:")
            print(f"   Language Switch: {result['languageSwitch']}")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Churn Risk: {result['userChurnRisk']}")
            print(f"   Task Completion: {result['taskCompletion']}")
        else:
            print(f"❌ {result['filename']}: {result['error_message']}")
        
        # Add delay between requests to be respectful to API
        if i < total_files:
            time.sleep(2)
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n📊 Processing complete: {successful}/{total_files} successful in {total_time:.1f} seconds")
    
    return results

def save_to_csv(results: List[Dict[str, Any]], csv_path: str):
    """Save LLM evaluation results to CSV"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"✅ Results saved to: {csv_path}")

def print_summary(results: List[Dict[str, Any]]):
    """Print LLM evaluation summary"""
    if not results:
        print("📊 No results to summarize")
        return
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"\n📊 LLM EVALUATION SUMMARY")
    print(f"=" * 50)
    print(f"Total Files: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        # LLM Evaluation metrics
        language_switch_count = sum(1 for r in successful if r.get('languageSwitch'))
        sentiment_counts = {}
        for r in successful:
            sentiment = r.get('sentiment')
            if sentiment:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Behavioral Analysis metrics
        churn_risk_count = sum(1 for r in successful if r.get('userChurnRisk'))
        user_repetition_count = sum(1 for r in successful if r.get('userRepetition'))
        agent_repetition_count = sum(1 for r in successful if r.get('agentRepetition'))
        
        # Task Completion metrics
        task_completion_counts = {}
        for r in successful:
            task_completion = r.get('taskCompletion')
            if task_completion:
                task_completion_counts[task_completion] = task_completion_counts.get(task_completion, 0) + 1
        
        avg_processing_time = sum(r.get('processing_time_seconds', 0) for r in successful) / len(successful)
        
        print(f"\n🗣️ LLM EVALUATION RESULTS:")
        print(f"🌐 Language Switch: {language_switch_count}/{len(successful)} files ({language_switch_count/len(successful)*100:.1f}%)")
        
        print(f"😊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = count/len(successful)*100
            print(f"   {sentiment}: {count} files ({percentage:.1f}%)")
        
        print(f"\n🧠 BEHAVIORAL ANALYSIS RESULTS:")
        print(f"🚨 Churn Risk Detected: {churn_risk_count}/{len(successful)} files ({churn_risk_count/len(successful)*100:.1f}%)")
        print(f"🔄 User Repetition: {user_repetition_count}/{len(successful)} files ({user_repetition_count/len(successful)*100:.1f}%)")
        print(f"🤖 Agent Repetition: {agent_repetition_count}/{len(successful)} files ({agent_repetition_count/len(successful)*100:.1f}%)")
        
        print(f"\n✅ TASK COMPLETION RESULTS:")
        for completion_status, count in task_completion_counts.items():
            percentage = count/len(successful)*100
            print(f"   {completion_status}: {count} files ({percentage:.1f}%)")
        
        print(f"\n⏱️  Average Processing Time: {avg_processing_time:.1f} seconds per file")
    
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
    print("🧠 Fast LLM Logger - Complete Audio Analysis")
    print("=" * 50)
    
    # Setup
    ensure_output_directory()
    csv_output_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    
    # Get audio files
    print(f"\n📁 Looking for audio files...")
    audio_files = get_audio_files_from_csv(INPUT_CSV, AUDIO_FILES_DIR)
    
    if not audio_files:
        print("❌ No audio files found to process")
        return
    
    print("✅ Complete LLM evaluation enabled (language, sentiment, behavioral analysis, task completion)")
    
    # Process files for complete LLM evaluation
    results = process_batch_llm_evaluation(audio_files)
    
    if results:
        # Save to CSV
        save_to_csv(results, csv_output_path)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ LLM evaluation complete!")
        print(f"📄 CSV file: {os.path.abspath(csv_output_path)}")
        print(f"💡 Complete analysis: Language switch, sentiment, churn risk, repetition, and task completion")
    else:
        print("❌ No results to save")

if __name__ == "__main__":
    main()
