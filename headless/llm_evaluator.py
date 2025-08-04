import os
import requests
import yaml
import time
import traceback
import json
import re
import base64
import mimetypes
import httpx
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional
from dotenv import load_dotenv
from schemas import BehavioralAnalysisResult

load_dotenv()

class LlmEvaluationResult(BaseModel):
    languageSwitch: bool = Field(..., description="Whether the agent switched languages.")
    sentiment: Literal["happy", "neutral", "angry", "disappointed"] = Field(..., description="The user's sentiment.")

class CombinedEvaluationResult(BaseModel):
    """Combined result containing both LLM evaluation and behavioral analysis"""
    # LLM Evaluation results
    languageSwitch: bool = Field(..., description="Whether the agent switched languages.")
    sentiment: Literal["happy", "neutral", "angry", "disappointed"] = Field(..., description="The user's sentiment.")
    # Behavioral Analysis results
    negativeExperience: bool = Field(..., description="Whether the user shows negative experience indicators")
    negativeExperienceReasoning: Optional[str] = Field(None, description="1-2 short sentences explaining the negative experience assessment")
    userRepetition: bool = Field(..., description="Whether user shows problematic repetitive behavior")
    agentRepetition: bool = Field(..., description="Whether agent shows problematic repetitive behavior")
    taskCompletion: Literal["Fully Completed", "Partially Completed", "Not Completed"] = Field(..., description="Whether the user achieved their primary goal by the end of the call")
    taskCompletionReasoning: str = Field(..., description="One-sentence justification for the task completion assessment")

class LlmEvaluator:
    def __init__(self):
        # Check for required API keys (only Gemini proxy now)
        self.gemini_api_key = os.getenv("PROXY_API_KEY")
        self.gemini_proxy_base_url = os.getenv("GEMINI_PROXY_BASE_URL")
        
        if not self.gemini_api_key:
            raise ValueError("PROXY_API_KEY not found in environment variables")
        if not self.gemini_proxy_base_url:
            raise ValueError("GEMINI_PROXY_BASE_URL not found in environment variables")
            
        print(f"✅ Initializing LLM Evaluator with Gemini proxy configuration")
        
        # Load prompts from YAML file
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load evaluation prompts from YAML file."""
        prompts_file = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_file, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
            print("✅ Evaluation prompts loaded from YAML file")
            return prompts

    def _get_mime_type(self, file_path: str) -> str:
        """Get the MIME type for an audio file."""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Common audio MIME types
        audio_mime_types = {
            '.mp3': 'audio/mp3',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac'
        }
        
        if mime_type and mime_type.startswith('audio/'):
            return mime_type
        
        # Fallback to extension-based detection
        ext = os.path.splitext(file_path)[1].lower()
        return audio_mime_types.get(ext, 'audio/mp3')  # Default to mp3

    def _encode_audio_file(self, file_path: str) -> tuple[str, str]:
        """Read and base64 encode an audio file. Returns (base64_data, mime_type)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        base64_data = base64.b64encode(audio_bytes).decode('utf-8')
        mime_type = self._get_mime_type(file_path)
        
        return base64_data, mime_type

    def _get_combined_response_schema(self) -> dict:
        """Get the response schema for complete semantic analysis (LLM + behavioral) structured output."""
        return {
            "type": "OBJECT",
            "properties": {
                # LLM Evaluation fields
                "languageSwitch": {
                    "type": "BOOLEAN",
                    "description": "Whether the agent switched languages during the conversation"
                },
                "sentiment": {
                    "type": "STRING",
                    "enum": ["happy", "neutral", "angry", "disappointed"],
                    "description": "The user's overall sentiment during the conversation"
                },
                # Behavioral Analysis fields
                "negativeExperience": {
                    "type": "BOOLEAN",
                    "description": "Whether the user shows explicit expressions of strong dissatisfaction that indicate significant service quality issues"
                },
                "negativeExperienceReasoning": {
                    "type": "STRING",
                    "description": "1-2 short sentences explaining the specific negative experience indicators (null if no negative experience)"
                },
                "userRepetition": {
                    "type": "BOOLEAN",
                    "description": "Whether user repeats reasonable requests 3+ times because agent fails to address them appropriately"
                },
                "agentRepetition": {
                    "type": "BOOLEAN", 
                    "description": "Whether agent shows 3+ instances of identical responses that fail to advance conversation meaningfully"
                },
                "taskCompletion": {
                    "type": "STRING",
                    "enum": ["Fully Completed", "Partially Completed", "Not Completed"],
                    "description": "Whether the user achieved their primary goal by the end of the call"
                },
                "taskCompletionReasoning": {
                    "type": "STRING",
                    "description": "One-sentence justification for the task completion assessment"
                }
            },
            "required": ["languageSwitch", "sentiment", "negativeExperience", "userRepetition", "agentRepetition", "taskCompletion", "taskCompletionReasoning"],
            "propertyOrdering": ["languageSwitch", "sentiment", "negativeExperience", "negativeExperienceReasoning", "userRepetition", "agentRepetition", "taskCompletion", "taskCompletionReasoning"]
        }

    def _call_gemini_via_proxy(self, prompt: str, audio_file_path: str = None, response_schema: dict = None) -> str:
        """Call Gemini API through proxy server with audio support and optional structured output."""
        if not self.gemini_proxy_base_url or not self.gemini_api_key:
            raise ValueError("Gemini proxy configuration not available")

        # The full URL to the proxy's generation endpoint
        model_name = "gemini-2.5-flash"  # or "gemini-2.0-pro"
        url = f"{self.gemini_proxy_base_url}/{model_name}:generateContent"

        # Headers required by the proxy
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.gemini_api_key}",
        }

        # Get system instruction from prompts.yml
        system_instruction = self.prompts.get("system_prompt", "")

        # Build the parts list for the user message
        parts = []
        
        # Add user prompt text
        if prompt:
            parts.append({"text": prompt})
        
        # Add audio file if provided
        if audio_file_path and os.path.exists(audio_file_path):
            try:
                base64_data, mime_type = self._encode_audio_file(audio_file_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_data
                    }
                })
                print(f"Added audio file: {audio_file_path} ({mime_type})")
            except Exception as e:
                print(f"Error processing audio file: {e}")
                raise

        if not parts:
            raise ValueError("No content to send (no prompt or valid audio file)")

        # Build generation config
        generation_config = {
            "temperature": 0.7
        }
        
        # Add structured output configuration if schema provided
        if response_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema

        # Payload structured for the Gemini API
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": generation_config,
        }
        
        # Add system instruction from prompts.yml
        if system_instruction:
            payload["system_instruction"] = {
                "parts": [
                    {
                        "text": system_instruction
                    }
                ]
            }

        print(f"Sending request to: {url}")
        if audio_file_path:
            print(f"Including audio file: {audio_file_path}")
        if response_schema:
            print("Using structured output with responseSchema")
        if system_instruction:
            print("Using system instruction from prompts.yml")

        try:
            # Using a timeout for audio processing
            with httpx.Client(timeout=120.0) as client:  # Increased timeout for audio processing
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                # Parse the JSON response from the API
                response_data = response.json()
                
                # Extract the text from the response
                text_content = response_data['candidates'][0]['content']['parts'][0]['text']
                
                return text_content

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error occurred: {e.response.status_code} - {e.response.reason_phrase}")
            print("Response body:", e.response.text)
            raise
        except (KeyError, IndexError) as e:
            print("Error: Could not parse the response from the API.")
            print("Full response received:", response.text if 'response' in locals() else "No response received")
            print(f"Parse error: {e}")
            raise
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            print(f"Error details: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def gemini_semantic_analysis(self, audio_file_path: str, language: str = "English") -> Optional[CombinedEvaluationResult]:
        """
        Perform complete semantic analysis using Gemini multimodal (LLM + behavioral analysis in one call)
        
        Args:
            audio_file_path: Path to the audio file to analyze
            language: Expected language for the conversation
            
        Returns:
            CombinedEvaluationResult or None if analysis fails
        """
        try:
            print(f"\n--- Starting Gemini Semantic Analysis for {os.path.basename(audio_file_path)} ---")
            
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Create the combined prompt for complete analysis
            combined_prompt = self._create_combined_prompt(language)
            
            # Get the response schema for structured output
            response_schema = self._get_combined_response_schema()
            
            # Generate analysis using proxy-based Gemini with structured output
            print("🧠 Generating complete semantic analysis with structured output...")
            response_text = self._call_gemini_via_proxy(combined_prompt, audio_file_path, response_schema)
            
            print(f"📄 Raw response: {response_text}")
            
            # Parse the JSON response (guaranteed to be valid JSON due to responseSchema)
            try:
                response_json = json.loads(response_text)
                
                # Create result object directly from the structured response
                result = CombinedEvaluationResult(
                    languageSwitch=response_json.get('languageSwitch', False),
                    sentiment=response_json.get('sentiment', 'neutral'),
                    negativeExperience=response_json.get('negativeExperience', False),
                    negativeExperienceReasoning=response_json.get('negativeExperienceReasoning'),
                    userRepetition=response_json.get('userRepetition', False),
                    agentRepetition=response_json.get('agentRepetition', False),
                    taskCompletion=response_json.get('taskCompletion', 'Not Completed'),
                    taskCompletionReasoning=response_json.get('taskCompletionReasoning', 'No reasoning provided')
                )
                
                print("✅ Gemini semantic analysis completed successfully:")
                print(f"  - Language Switch: {result.languageSwitch}")
                print(f"  - Sentiment: {result.sentiment}")
                print(f"  - Negative Experience: {result.negativeExperience}")
                if result.negativeExperienceReasoning:
                    print(f"  - Negative Experience Reasoning: {result.negativeExperienceReasoning}")
                print(f"  - User Repetition: {result.userRepetition}")
                print(f"  - Agent Repetition: {result.agentRepetition}")
                print(f"  - Task Completion: {result.taskCompletion}")
                print(f"  - Task Completion Reasoning: {result.taskCompletionReasoning}")
                
                return result
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ Failed to parse semantic analysis response: {e}")
                print(f"Raw response: {response_text}")
                return None
                
        except Exception as e:
            print(f"❌ Error in Gemini semantic analysis: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None

    def _create_combined_prompt(self, language: str = "English") -> str:
        """Create the combined prompt for complete semantic analysis from YAML templates"""
        # Use the unified behavioral_task_prompt from YAML
        behavioral_task_prompt = self.prompts.get("behavioral_task_prompt", "")
        
        # Format the template with provided values
        formatted_prompt = behavioral_task_prompt.format(
            language=language
        )
        
        # Note: No need to add JSON schema instructions since we're using responseSchema
        # for structured output which guarantees valid JSON responses
        
        return formatted_prompt

    def run_combined_evaluation(self, file_path: str, language: str = "English") -> CombinedEvaluationResult:
        """
        Runs complete semantic analysis using Gemini for a given audio file.
        """
        print(f"\n===== Starting Combined Semantic Evaluation for {os.path.basename(file_path)} =====")
        
        # Run complete semantic analysis using Gemini
        result = self.gemini_semantic_analysis(file_path, language)
        
        if not result:
            print("❌ Semantic analysis failed. Using default values.")
            # Return default values if analysis fails
            result = CombinedEvaluationResult(
                languageSwitch=False,
                sentiment="null",
                negativeExperience=False,
                negativeExperienceReasoning=None,
                userRepetition=False,
                agentRepetition=False,
                taskCompletion="Not Completed",
                taskCompletionReasoning="Analysis failed or was unavailable"
            )
        
        print(f"===== Combined Semantic Evaluation for {os.path.basename(file_path)} Complete =====\n")
        return result

    def run_evaluation(self, file_path: str, language: str = "English") -> LlmEvaluationResult:
        """
        Runs basic LLM evaluation using Gemini for a given audio file.
        
        Note: This method extracts only LLM evaluation fields from the complete analysis.
        For full analysis, use run_combined_evaluation() instead.
        """
        print(f"\n===== Starting Basic LLM Evaluation for {os.path.basename(file_path)} =====")
        
        # Run complete semantic analysis and extract LLM parts
        combined_result = self.gemini_semantic_analysis(file_path, language)
        
        if not combined_result:
            print("❌ Semantic analysis failed. Using default values.")
            # Return default values if analysis fails
            basic_result = LlmEvaluationResult(
                languageSwitch=False,
                sentiment="null"
            )
        else:
            # Extract only LLM evaluation fields
            basic_result = LlmEvaluationResult(
                languageSwitch=combined_result.languageSwitch,
                sentiment=combined_result.sentiment
            )
        
        print(f"===== Basic LLM Evaluation for {os.path.basename(file_path)} Complete =====\n")
        return basic_result

    def analyze_batch_semantic(self, audio_files: list[str], language: str = "English") -> Dict[str, Optional[CombinedEvaluationResult]]:
        """
        Analyze multiple audio files for complete semantic analysis using Gemini
        
        Args:
            audio_files: List of audio file paths
            language: Expected language for the conversation
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        results = {}
        total_files = len(audio_files)
        
        print(f"\n🚀 Starting batch semantic analysis for {total_files} files")
        
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n📊 Processing file {i}/{total_files}: {os.path.basename(file_path)}")
            
            try:
                result = self.gemini_semantic_analysis(file_path, language)
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

