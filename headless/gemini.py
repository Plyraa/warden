import httpx
import json
import os
from dotenv import load_dotenv
# --- CONFIGURATION: FILL IN YOUR DETAILS HERE ---

# 3. The specific Gemini model you want to use
MODEL_NAME = "gemini-2.0-pro" # Or "gemini-1.5-pro-latest", etc.

# 4. The prompt you want to send to the AI
USER_PROMPT = "Explain how AI works in a few words"

# --- END OF CONFIGURATION ---
load_dotenv = True  # Set to True if you want to load environment variables from a .env file
if load_dotenv:
    PROXY_API_KEY = os.getenv("PROXY_API_KEY")
    PROXY_BASE_URL = os.getenv("GEMINI_PROXY_BASE_URL")
    
def call_gemini_via_proxy(prompt: str):
    """
    Calls the Gemini API through a custom proxy server.
    """
    if not PROXY_BASE_URL or "YOUR_PROXY" in PROXY_BASE_URL:
        print("Error: Please set your PROXY_BASE_URL at the top of the script.")
        return
    if not PROXY_API_KEY or "YOUR_PROXY" in PROXY_API_KEY:
        print("Error: Please set your PROXY_API_KEY at the top of the script.")
        return

    # The full URL to the proxy's generation endpoint
    # This follows the structure from your example.
    # We use :generateContent for a simple, non-streaming response.
    url = f"{PROXY_BASE_URL}/{MODEL_NAME}:generateContent"

    # Headers required by your proxy
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PROXY_API_KEY}",
    }

    # Payload structured for the Gemini API
    # 'contents' is a list representing the conversation history.
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generation_config": {
            "temperature": 0.7
        },
    }

    print(f"Sending request to: {url}")
    print("-" * 20)

    try:
        # Using a timeout is always a good practice
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)

            # Raise an exception if the request failed (e.g., 4xx or 5xx error)
            response.raise_for_status()

            # Parse the JSON response from the API
            response_data = response.json()
            
            # Extract the text from the response
            # The structure is typically nested like this.
            # We add error handling in case the structure is different.
            text_content = response_data['candidates'][0]['content']['parts'][0]['text']
            
            return text_content

    except httpx.HTTPStatusError as e:
        # This catches errors like 401 Unauthorized, 404 Not Found, 500 Server Error
        print(f"HTTP Error occurred: {e.response.status_code} - {e.response.reason_phrase}")
        print("Response body:", e.response.text)
    except (KeyError, IndexError):
        print("Error: Could not parse the response from the API.")
        print("Full response received:", response.text)
    except httpx.RequestError as e:
        # This catches network errors like connection refused, timeout, etc.
        print(f"An error occurred while requesting {e.request.url!r}.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None

# --- Main execution ---
if __name__ == "__main__":
    ai_response = call_gemini_via_proxy(USER_PROMPT)
    
    if ai_response:
        print("\n--- Gemini's Response ---")
        print(ai_response)
        print("-------------------------\n")