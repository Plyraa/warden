import os
import requests
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from dotenv import load_dotenv
import json

load_dotenv()

class LlmEvaluationResult(BaseModel):
    toneAdherence: int = Field(..., description="Adherence to the specified tone, from 1 to 5.", ge=1, le=5)
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
        system_prompt = """
        You are an expert in conversation analysis. Your task is to evaluate a conversation between a user and an AI agent based on a provided transcript and the agent's predefined properties. 
        You must provide your evaluation in a structured format.
        """

        user_prompt = f"""
        Please evaluate the following conversation based on the provided agent properties.

        **Agent Properties:**
        - **Tone:** {agent_properties.get('tone')}
        - **Persona:** {agent_properties.get('optimizedPersona')}
        - **Language:** {agent_properties.get('language')}
        - **Role:** {agent_properties.get('role')}

        **Conversation Transcript:**
        Note: The transcript is formatted with speaker labels (User/Agent) and may contain multiple turns. But those labels may be missing or plain wrong, like Agent message labeled as User or User message labeled as Agent. Infer what messages are from User and Agent yourself.
        {transcript}

        **Evaluation Criteria:**

        1.  **Tone Adherence (Score 1-5):**
            - Evaluate if the agent's responses match the specified tone and persona.
            - 1: Not at all; 5: Perfectly.

        2.  **Persona Adherence (Score 1-5):**
            - Evaluate if the agent's messages align with its persona.
            - 1: Not at all; 5: Perfectly.

        3.  **Language Switch (Boolean):**
            - Determine if the agent switched from the specified language ('{agent_properties.get('language')}').
            - `true` if a switch occurred, `false` otherwise.

        4.  **User Sentiment (Enum: "happy", "neutral", "angry", "disappointed"):**
            - Analyze the user's overall sentiment towards the AI agent.
            - Consider tone, word choice, and context.
            - **Happy:** Satisfaction, gratitude.
            - **Angry:** Frustration, harsh language.
            - **Neutral:** Matter-of-fact, no strong emotion.
            - **Disappointed:** Letdown, unfulfilled expectations.
            - Briefly explain your reasoning before providing the final sentiment.
        """

        print("... Sending the following prompt to OpenAI ...")

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
        print(f"toneAdherence: {parsed_result.toneAdherence}")
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

