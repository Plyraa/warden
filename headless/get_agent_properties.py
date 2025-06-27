import aiohttp
import json
import asyncio

async def get_agent_properties(agent_id):
    headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": f"https://www.jotform.com",
            "Referer": f"https://www.jotform.com/agent/{agent_id}/phone",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Not;A=Brand";v="24", "Chromium";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        }
    headers["Referer"] = f"https://www.jotform.com/agent/{agent_id}/phone"
    gender = ""
    language = ""
    persona = ""
    voice_id = None
    message_response = None
    jf_trace_id = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://www.jotform.com/API/ai-agent-builder/agents/{agent_id}/properties",
                headers=headers,
                timeout=15,
            ) as message_response:
                jf_trace_id = message_response.headers.get("jf-trace-id", None) if message_response and message_response.headers else None
                # Raise for status
                message_response.raise_for_status()

                # Read response content
                content = await message_response.content.read()
                response = json.loads(content.decode())
                return response
                
    except Exception as e:
        print(f"Unexpected error occurred while processing agent's tasks: {str(e)}, agent_id: {agent_id}")
        return None

async def main() -> None:
    """Fetch agent properties and save them to a JSON file."""

    agent_id = "0197362dee337c83853df36020378b3390f8"
    response = await get_agent_properties(agent_id)

    if response is None:
        print("No response received; nothing to save.")
        return

    filename = f"{agent_id}_properties.json"
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(response, file, indent=2, ensure_ascii=False)
        print(f"Saved agent properties to {filename}")
    except IOError as io_err:
        print(f"Failed to write JSON to {filename}: {io_err}")


if __name__ == "__main__":
    asyncio.run(main())