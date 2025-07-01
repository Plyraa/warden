import os
import requests
import yaml
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from dotenv import load_dotenv
import json

load_dotenv()

class LlmEvaluationResult(BaseModel):
    personaAdherence: int = Field(..., description="Adherence to the specified persona, from 1 to 5.", ge=1, le=5)
    languageSwitch: bool = Field(..., description="Whether the agent switched languages.")
    sentiment: Literal["happy", "neutral", "angry", "disappointed"] = Field(..., description="The user's sentiment.")

class LlmEvaluator:
    def __init__(self):
        # Check for required API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
            
        print(f"✅ Initializing LLM Evaluator with API keys present")
        
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
        
        print("✅ Agent properties fetched successfully:")
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
        print("✅ Transcription complete:")
        print(full_transcript)
        return full_transcript

    def evaluate_transcript(self, transcript: str, agent_properties: Dict[str, Any]) -> LlmEvaluationResult:
        """
        Evaluates the transcript using OpenAI's gpt-4o model with structured output.
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
        
        print("✅ OpenAI evaluation successful. Received structured output:")
        #print(response.model_dump_json(indent=2))
        
        # Extract the parsed result from the response
        parsed_result = response.output[0].content[0].parsed
        print("✅ Extracted parsed result:")
        print(f"personaAdherence: {parsed_result.personaAdherence}")
        print(f"languageSwitch: {parsed_result.languageSwitch}")
        print(f"sentiment: {parsed_result.sentiment}")
        
        return parsed_result

    def run_evaluation(self, file_path: str, agent_id: str):
        """
        Runs the full evaluation pipeline for a given audio file and agent ID.
        """
        print(f"\n===== Starting LLM Evaluation for {os.path.basename(file_path)} =====")
        agent_properties = self.get_agent_properties(agent_id)
        transcript = self.transcribe_audio(file_path)
        if not transcript:
            print("❌ Transcription failed or produced empty text. Skipping evaluation.")
            raise ValueError("Transcription failed or produced empty text.")
        evaluation = self.evaluate_transcript(transcript, agent_properties)
        print(f"===== LLM Evaluation for {os.path.basename(file_path)} Complete =====\n")
        return evaluation

