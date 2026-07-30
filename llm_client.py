"""
LLM Client — single place to switch between backends.
Comment / uncomment the llm_call assignment at the bottom.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()


def call_ollama(prompt: str) -> str:
    """Local Ollama with llama3.2:3b."""
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"]


def call_genai(prompt: str) -> str:
    """GenAI Lab endpoint via LangChain ChatOpenAI."""
    from langchain_openai import ChatOpenAI

    ssl_verify = os.getenv("SSL_VERIFY", "True").strip().lower() == "true"
    http_client = httpx.Client(verify=ssl_verify)

    llm = ChatOpenAI(
        base_url=os.getenv("GENAI_LAB_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME", "azure_ai/genailab-maas-DeepSeek-V3-0324"),
        api_key=os.getenv("GENAI_LAB_API_KEY"),
        http_client=http_client,
        temperature=0.1,
    )
    response = llm.invoke(prompt)
    return response.content


# --- Choose your LLM Backend ---
# Comment / uncomment to switch:
llm_call = call_genai
# llm_call = call_genai
