"""
Gemini-based behavioral analysis for customer churn risk and repetition detection
"""
import os
import yaml
import time
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from schemas import BehavioralAnalysisResult

load_dotenv()

class GeminiBehavioralAnalyzer:
    def __init__(self):
        """Initialize the Gemini behavioral analyzer using Google GenAI client"""
        # Get the Gemini API key (can work with proxy)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.gemini_api_key)
        
        # Load prompts from YAML file
        self.prompts = self._load_prompts()
        print("✅ Gemini Behavioral Analyzer initialized with GenAI client")

    def _load_prompts(self) -> Dict[str, str]:
        """Load behavioral analysis prompts from YAML file"""
        prompts_file = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_file, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
            print("✅ Behavioral analysis prompts loaded from YAML file")
            return prompts

    def analyze_audio_file(self, audio_file_path: str) -> Optional[BehavioralAnalysisResult]:
        """
        Analyze an audio file for behavioral patterns using Gemini multimodal
        
        Args:
            audio_file_path: Path to the audio file to analyze
            
        Returns:
            BehavioralAnalysisResult or None if analysis fails
        """
        try:
            print(f"\n--- Starting Gemini Behavioral Analysis for {os.path.basename(audio_file_path)} ---")
            
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Upload audio file to Gemini (following your example)
            print("📤 Uploading audio file to Gemini...")
            uploaded_file = self.client.files.upload(file=audio_file_path)
            print(f"✅ File uploaded successfully: {uploaded_file.name}")
            
            # Wait a moment for file processing
            #time.sleep(2)
            
            # Create the prompt for behavioral analysis
            behavioral_prompt = self._create_behavioral_prompt()
            
            # Generate analysis using multimodal Gemini (following your example)
            print("🧠 Generating behavioral analysis...")
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",  # Using flash as in your example
                contents=[behavioral_prompt, uploaded_file]
            )
            
            print(f"📄 Raw response: {response.text}")
            
            # Parse the response
            try:
                import json
                # Try to extract JSON from the response
                response_text = response.text.strip()
                
                # Sometimes the response might have extra text, try to find JSON
                if response_text.startswith('{') and response_text.endswith('}'):
                    response_json = json.loads(response_text)
                else:
                    # Look for JSON within the response
                    import re
                    json_match = re.search(r'\{[^{}]*\}', response_text)
                    if json_match:
                        response_json = json.loads(json_match.group())
                    else:
                        print("❌ Could not find valid JSON in response")
                        return None
                
                # Create result object
                result = BehavioralAnalysisResult(
                    userChurnRisk=response_json.get('userChurnRisk', False),
                    userChurnReasoning=response_json.get('userChurnReasoning'),
                    userRepetition=response_json.get('userRepetition', False),
                    agentRepetition=response_json.get('agentRepetition', False)
                )
                
                print("✅ Gemini analysis completed successfully:")
                print(f"  - User Churn Risk: {result.userChurnRisk}")
                if result.userChurnReasoning:
                    print(f"  - Churn Reasoning: {result.userChurnReasoning}")
                print(f"  - User Repetition: {result.userRepetition}")
                print(f"  - Agent Repetition: {result.agentRepetition}")
                
                # Clean up uploaded file
                try:
                    self.client.files.delete(uploaded_file.name)
                    print("🗑️ Temporary file cleaned up")
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Could not clean up file {uploaded_file.name}: {cleanup_error}")
                
                return result
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ Failed to parse response: {e}")
                print(f"Raw response: {response.text}")
                
                # Clean up uploaded file even on error
                try:
                    self.client.files.delete(uploaded_file.name)
                    print("🗑️ Temporary file cleaned up")
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Could not clean up file {uploaded_file.name}: {cleanup_error}")
                
                return None
                
        except Exception as e:
            print(f"❌ Error in Gemini behavioral analysis: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None

    def _create_behavioral_prompt(self) -> str:
        """Create the behavioral analysis prompt from YAML templates"""
        # Use the behavioral analysis prompt from prompts.yaml
        behavioral_prompt = self.prompts.get("behavioral_analysis_prompt", "")
        behavioral_template = self.prompts.get("behavioral_analysis_template", "")
        
        # Combine the prompt and template
        full_prompt = f"{behavioral_prompt}\n\n{behavioral_template}"
        
        # Remove the transcript placeholder since we're working with audio directly
        full_prompt = full_prompt.replace("CONVERSATION TRANSCRIPT:\n{transcript}", 
                                         "AUDIO CONVERSATION:\nAnalyze the provided audio file for behavioral patterns.")
        
        # Add JSON schema requirements for structured output
        json_schema = """
        
        IMPORTANT: Return your analysis in valid JSON format matching this exact schema:
        {
            "userChurnRisk": boolean,
            "userChurnReasoning": "string (1-2 short sentences if userChurnRisk is true, null otherwise)",
            "userRepetition": boolean,
            "agentRepetition": boolean
        }
        
        Ensure the response is valid JSON only, without any additional text, formatting, or code blocks.
        """
        
        full_prompt += json_schema
        
        return full_prompt

    def analyze_batch(self, audio_files: list[str]) -> Dict[str, Optional[BehavioralAnalysisResult]]:
        """
        Analyze multiple audio files for behavioral patterns
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        results = {}
        total_files = len(audio_files)
        
        print(f"\n🚀 Starting batch behavioral analysis for {total_files} files")
        
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n📊 Processing file {i}/{total_files}: {os.path.basename(file_path)}")
            
            try:
                result = self.analyze_audio_file(file_path)
                results[file_path] = result
                
                if result:
                    print(f"✅ Analysis complete for {os.path.basename(file_path)}")
                else:
                    print(f"❌ Analysis failed for {os.path.basename(file_path)}")
                    
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                results[file_path] = None
            
            # Add a small delay between requests to be respectful to the API
            if i < total_files:
                time.sleep(2)
        
        successful = sum(1 for result in results.values() if result is not None)
        print(f"\n📊 Batch analysis complete: {successful}/{total_files} successful")
        
        return results

def test_analyzer():
    """Test function for the behavioral analyzer"""
    try:
        print("🧪 Testing Gemini Behavioral Analyzer with multimodal audio analysis")
        print("=" * 70)
        
        analyzer = GeminiBehavioralAnalyzer()
        
        # Test with sample audio files
        test_files = [
            r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls\test1.mp3",
            r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls\66057328368247d623d0b87.67876133.mp3"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n🎵 Testing with: {os.path.basename(test_file)}")
                
                # Check file size
                file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
                print(f"📊 File size: {file_size_mb:.2f} MB")
                
                if file_size_mb > 20:
                    print("⚠️ Warning: File is larger than 20MB, may hit upload limits")
                
                print("🎯 Using actual multimodal analysis with file upload!")
                
                result = analyzer.analyze_audio_file(test_file)
                
                if result:
                    print("🎉 Test successful!")
                    print(f"  - User Churn Risk: {result.userChurnRisk}")
                    if result.userChurnReasoning:
                        print(f"  - Churn Reasoning: {result.userChurnReasoning}")
                    print(f"  - User Repetition: {result.userRepetition}")
                    print(f"  - Agent Repetition: {result.agentRepetition}")
                    break
                else:
                    print("❌ Analysis failed for this file")
            else:
                print(f"⚠️ Test file not found: {test_file}")
        
        print("\n✅ Test completed!")
        print("🎵 The analyzer is now using actual multimodal audio analysis!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    test_analyzer()
