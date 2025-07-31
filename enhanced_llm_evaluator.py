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

load_dotenv()

class LlmEvaluationResult(BaseModel):
    personaAdherence: int = Field(..., description="Adherence to the specified persona, from 1 to 5.", ge=1, le=5)
    languageSwitch: bool = Field(..., description="Whether the agent switched languages.")
    sentiment: Literal["happy", "neutral", "angry", "disappointed"] = Field(..., description="The user's sentiment.")

class BehavioralAnalysisResult(BaseModel):
    """Model for behavioral analysis results from Gemini"""
    userChurnRisk: bool = Field(..., description="Whether the customer shows genuine churn risk indicators")
    userChurnReasoning: Optional[str] = Field(None, description="1-2 short sentences explaining the churn risk assessment")
    userRepetition: bool = Field(..., description="Whether user shows problematic repetitive behavior")
    agentRepetition: bool = Field(..., description="Whether agent shows problematic repetitive behavior")
    taskCompletion: Literal["Fully Completed", "Partially Completed", "Not Completed"] = Field(..., description="Whether the user achieved their primary goal by the end of the call")
    taskCompletionReasoning: str = Field(..., description="One-sentence justification for the task completion assessment")

class CombinedEvaluationResult(BaseModel):
    """Combined result containing both LLM evaluation and behavioral analysis"""
    # LLM Evaluation results (OpenAI-based)
    personaAdherence: int = Field(..., description="Adherence to the specified persona, from 1 to 5.", ge=1, le=5)
    languageSwitch: bool = Field(..., description="Whether the agent switched languages.")
    sentiment: Literal["happy", "neutral", "angry", "disappointed"] = Field(..., description="The user's sentiment.")
    # Behavioral Analysis results (Gemini-based)
    userChurnRisk: bool = Field(..., description="Whether the customer shows genuine churn risk indicators")
    userChurnReasoning: Optional[str] = Field(None, description="1-2 short sentences explaining the churn risk assessment")
    userRepetition: bool = Field(..., description="Whether user shows problematic repetitive behavior")
    agentRepetition: bool = Field(..., description="Whether agent shows problematic repetitive behavior")
    taskCompletion: Literal["Fully Completed", "Partially Completed", "Not Completed"] = Field(..., description="Whether the user achieved their primary goal by the end of the call")
    taskCompletionReasoning: str = Field(..., description="One-sentence justification for the task completion assessment")

class EnhancedLlmEvaluator:
    def __init__(self):
        # Check for required API keys
        # OpenAI for basic LLM evaluation
        openai_key = os.getenv("OPENAI_API_KEY")
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Gemini proxy for behavioral analysis
        self.gemini_api_key = os.getenv("PROXY_API_KEY")
        self.gemini_proxy_base_url = os.getenv("GEMINI_PROXY_BASE_URL")
        
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("PROXY_API_KEY not found in environment variables")
        if not self.gemini_proxy_base_url:
            raise ValueError("GEMINI_PROXY_BASE_URL not found in environment variables")
            
        print(f"✅ Initializing Enhanced LLM Evaluator with OpenAI + Gemini configuration")
        
        # Initialize OpenAI client for basic evaluation
        from openai import OpenAI
        from elevenlabs.client import ElevenLabs
        self.openai_client = OpenAI(base_url="https://dev.jotform.ai/openai/v1", api_key=openai_key)
        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
        self.jotform_agent_api_url = "https://www.jotform.com/API/ai-agent-builder/agents/{agent_id}/properties"
        
        # Load prompts from YAML file
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load evaluation prompts from YAML file."""
        prompts_file = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_file, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
            print("✅ Evaluation prompts loaded from YAML file")
            return prompts

    def get_agent_properties(self, agent_id: str) -> Dict[str, Any]:
        """
        Fetches agent properties from the Jotform API.
        """
        print(f"\n--- Step 1: Fetching Agent Properties for agent_id: {agent_id} ---")
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": "https://www.jotform.com",
            "Referer": f"https://www.jotform.com/agent/{agent_id}/phone",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Not;A=Brand";v="24", "Chromium";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        }
        url = self.jotform_agent_api_url.format(agent_id=agent_id)
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        properties_list = response.json().get("content", [])
        agent_properties = {}
        for prop in properties_list:
            agent_properties[prop['prop']] = prop['value']
        
        print("Agent properties fetched successfully:")
        print(json.dumps(agent_properties, indent=2))
        return agent_properties

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes audio using ElevenLabs and returns a formatted transcript.
        """
        print(f"\n--- Step 2: Transcribing Audio File: {os.path.basename(file_path)} ---")
        with open(file_path, "rb") as f:
            response = self.elevenlabs_client.speech_to_text.convert(
                file=f,
                model_id="scribe_v1_experimental",
                diarize=True,
                num_speakers=2,
                timestamps_granularity="word",
                tag_audio_events=False,
            )

        if not response.words:
            return ""
            
        speaker_ids = sorted(list(set(word.speaker_id for word in response.words if word.speaker_id is not None)))
        if not speaker_ids:
            # If no speaker IDs, just concatenate text.
            return response.text

        # Simple assumption: first speaker is agent. This might need refinement.
        agent_speaker_id = speaker_ids[0]
        user_speaker_id = speaker_ids[1] if len(speaker_ids) > 1 else speaker_ids[0]

        transcript = []
        current_speaker = None
        current_utterance = []

        for word in response.words:
            speaker = "Agent" if word.speaker_id == agent_speaker_id else "User"
            if current_speaker is None:
                current_speaker = speaker

            if speaker != current_speaker:
                transcript.append(f"{current_speaker}: {''.join(current_utterance)}")
                current_utterance = []
                current_speaker = speaker
            
            current_utterance.append(word.text)

        if current_utterance:
            transcript.append(f"{current_speaker}: {''.join(current_utterance)}")

        full_transcript = "\n".join(transcript)
        print("Transcription complete:")
        print(full_transcript)
        return full_transcript

    def evaluate_transcript(self, transcript: str, agent_properties: Dict[str, Any]) -> LlmEvaluationResult:
        """
        Evaluates the transcript using OpenAI's gpt-4.1-mini model with structured output.
        """
        print("\n--- Step 3: Evaluating Transcript with OpenAI ---")
        
        # Use prompts from YAML file
        system_prompt = self.prompts["system_prompt"]
        
        user_prompt = self.prompts["user_prompt_template"].format(
            persona=agent_properties.get('optimizedPersona', 'Not specified'),
            language=agent_properties.get('language', 'Not specified'),
            role=agent_properties.get('role', 'Not specified'),
            transcript=transcript
        )

        print("... Sending evaluation request to OpenAI ...")

        response = self.openai_client.responses.parse(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=LlmEvaluationResult
        )
        
        print("OpenAI evaluation successful. Received structured output:")
        
        # Extract the parsed result from the response
        parsed_result = response.output[0].content[0].parsed
        print("Extracted parsed result:")
        print(f"personaAdherence: {parsed_result.personaAdherence}")
        print(f"languageSwitch: {parsed_result.languageSwitch}")
        print(f"sentiment: {parsed_result.sentiment}")
        
        return parsed_result

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

    def _get_behavioral_response_schema(self) -> dict:
        """Get the response schema for behavioral analysis structured output."""
        return {
            "type": "OBJECT",
            "properties": {
                "userChurnRisk": {
                    "type": "BOOLEAN",
                    "description": "Whether the customer shows EXPLICIT dissatisfaction with service AND clear intent to stop using it (requires both harsh criticism AND intent to leave)"
                },
                "userChurnReasoning": {
                    "type": "STRING",
                    "description": "1-2 short sentences explaining the specific churn indicators (null if no churn risk)"
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
            "required": ["userChurnRisk", "userRepetition", "agentRepetition", "taskCompletion", "taskCompletionReasoning"],
            "propertyOrdering": ["userChurnRisk", "userChurnReasoning", "userRepetition", "agentRepetition", "taskCompletion", "taskCompletionReasoning"]
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
        
        # Add system instruction from prompts.yml if available
        system_instruction = self.prompts.get("system_prompt_behavioral", "")
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
            print("Using system instruction for behavioral analysis")

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

    def _create_behavioral_prompt(self, language: str = "English", role: str = "Assistant") -> str:
        """Create the behavioral analysis prompt from YAML templates"""
        behavioral_template = self.prompts.get("behavioral_analysis_template", "")
        
        # Create behavioral analysis prompt
        behavioral_prompt = f"""
BEHAVIORAL ANALYSIS TASK:
Analyze the provided audio conversation for behavioral indicators including user churn risk, repetitive patterns, and task completion.

AGENT SPECIFICATIONS:
- Required Language: {language}
- Agent Role: {role}

{behavioral_template}

IMPORTANT NOTES:
- Analyze the actual audio conversation directly
- Use audio context and conversation flow to identify who is speaking
- Consider the entire conversation flow, not isolated statements
- Focus on behavioral patterns and task resolution outcomes
"""
        
        # Format the template with provided values
        formatted_prompt = behavioral_prompt.format(
            language=language,
            role=role
        )
        
        return formatted_prompt

    def gemini_behavioral_analysis(self, audio_file_path: str, language: str = "English", role: str = "Assistant") -> Optional[BehavioralAnalysisResult]:
        """
        Perform behavioral analysis using Gemini multimodal
        
        Args:
            audio_file_path: Path to the audio file to analyze
            language: Expected language for the conversation
            role: Agent's role description
            
        Returns:
            BehavioralAnalysisResult or None if analysis fails
        """
        try:
            print(f"\n--- Starting Gemini Behavioral Analysis for {os.path.basename(audio_file_path)} ---")
            
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Create the behavioral analysis prompt
            behavioral_prompt = self._create_behavioral_prompt(language, role)
            
            # Get the response schema for structured output
            response_schema = self._get_behavioral_response_schema()
            
            # Generate analysis using proxy-based Gemini with structured output
            print("🧠 Generating behavioral analysis with structured output...")
            response_text = self._call_gemini_via_proxy(behavioral_prompt, audio_file_path, response_schema)
            
            print(f"📄 Raw response: {response_text}")
            
            # Parse the JSON response (guaranteed to be valid JSON due to responseSchema)
            try:
                response_json = json.loads(response_text)
                
                # Create result object directly from the structured response
                result = BehavioralAnalysisResult(
                    userChurnRisk=response_json.get('userChurnRisk', False),
                    userChurnReasoning=response_json.get('userChurnReasoning'),
                    userRepetition=response_json.get('userRepetition', False),
                    agentRepetition=response_json.get('agentRepetition', False),
                    taskCompletion=response_json.get('taskCompletion', 'Not Completed'),
                    taskCompletionReasoning=response_json.get('taskCompletionReasoning', 'No reasoning provided')
                )
                
                print("✅ Gemini behavioral analysis completed successfully:")
                print(f"  - User Churn Risk: {result.userChurnRisk}")
                if result.userChurnReasoning:
                    print(f"  - Churn Reasoning: {result.userChurnReasoning}")
                print(f"  - User Repetition: {result.userRepetition}")
                print(f"  - Agent Repetition: {result.agentRepetition}")
                print(f"  - Task Completion: {result.taskCompletion}")
                print(f"  - Task Completion Reasoning: {result.taskCompletionReasoning}")
                
                return result
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ Failed to parse behavioral analysis response: {e}")
                print(f"Raw response: {response_text}")
                return None
                
        except Exception as e:
            print(f"❌ Error in Gemini behavioral analysis: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None

    def run_combined_evaluation(self, file_path: str, agent_id: str) -> CombinedEvaluationResult:
        """
        Runs complete evaluation (OpenAI LLM + Gemini behavioral) for a given audio file.
        """
        print(f"\n===== Starting Combined Evaluation for {os.path.basename(file_path)} =====")
        
        # Step 1: Get agent properties and transcribe (for OpenAI evaluation)
        agent_properties = self.get_agent_properties(agent_id)
        transcript = self.transcribe_audio(file_path)
        
        # Step 2: Run OpenAI evaluation on transcript
        if not transcript:
            print("Transcription failed or produced empty text. Using default LLM values.")
            llm_result = LlmEvaluationResult(
                personaAdherence=3,
                languageSwitch=False,
                sentiment="neutral"
            )
        else:
            llm_result = self.evaluate_transcript(transcript, agent_properties)
        
        # Step 3: Run Gemini behavioral analysis on audio
        language = agent_properties.get('language', 'English')
        role = agent_properties.get('role', 'Assistant')
        behavioral_result = self.gemini_behavioral_analysis(file_path, language, role)
        
        if not behavioral_result:
            print("❌ Behavioral analysis failed. Using default values.")
            behavioral_result = BehavioralAnalysisResult(
                userChurnRisk=False,
                userChurnReasoning=None,
                userRepetition=False,
                agentRepetition=False,
                taskCompletion="Not Completed",
                taskCompletionReasoning="Analysis failed or was unavailable"
            )
        
        # Step 4: Combine results
        combined_result = CombinedEvaluationResult(
            personaAdherence=llm_result.personaAdherence,
            languageSwitch=llm_result.languageSwitch,
            sentiment=llm_result.sentiment,
            userChurnRisk=behavioral_result.userChurnRisk,
            userChurnReasoning=behavioral_result.userChurnReasoning,
            userRepetition=behavioral_result.userRepetition,
            agentRepetition=behavioral_result.agentRepetition,
            taskCompletion=behavioral_result.taskCompletion,
            taskCompletionReasoning=behavioral_result.taskCompletionReasoning
        )
        
        print(f"===== Combined Evaluation for {os.path.basename(file_path)} Complete =====\n")
        return combined_result

    def run_evaluation(self, file_path: str, agent_id: str) -> LlmEvaluationResult:
        """
        Runs basic LLM evaluation for backward compatibility.
        
        Note: For full analysis including behavioral metrics, use run_combined_evaluation() instead.
        """
        print(f"\n===== Starting Basic LLM Evaluation for {os.path.basename(file_path)} =====")
        
        agent_properties = self.get_agent_properties(agent_id)
        transcript = self.transcribe_audio(file_path)
        if not transcript:
            print("Transcription failed or produced empty text. Skipping evaluation.")
            raise ValueError("Transcription failed or produced empty text.")
        evaluation = self.evaluate_transcript(transcript, agent_properties)
        print(f"===== Basic LLM Evaluation for {os.path.basename(file_path)} Complete =====\n")
        return evaluation
