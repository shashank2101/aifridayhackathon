import os

def call_ollama(prompt: str) -> str:
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"]

# --- Choose your LLM Backend ---
llm_call = call_ollama
