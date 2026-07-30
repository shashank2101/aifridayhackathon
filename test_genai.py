"""
Test script for GenAI Lab LLM endpoint via LangChain.
Run:  python3 test_genai.py

If this works, you can switch llm_client.py to use call_genai instead of call_ollama.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()


def get_http_client() -> httpx.Client:
    ssl_verify = os.getenv("SSL_VERIFY", "True").strip().lower() == "true"
    return httpx.Client(verify=ssl_verify)


def get_llm_client():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=os.getenv("GENAI_LAB_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME", "azure_ai/genailab-maas-DeepSeek-V3-0324"),
        api_key=os.getenv("GENAI_LAB_API_KEY"),
        http_client=get_http_client(),
        temperature=0.1,
    )


if __name__ == "__main__":
    print("Testing GenAI Lab connection...")
    print(f"  Base URL : {os.getenv('GENAI_LAB_BASE_URL')}")
    print(f"  Model    : {os.getenv('LLM_MODEL_NAME', 'azure_ai/genailab-maas-DeepSeek-V3-0324')}")
    print(f"  API Key  : {'***' + os.getenv('GENAI_LAB_API_KEY', '')[-4:] if os.getenv('GENAI_LAB_API_KEY') else 'NOT SET'}")
    print()

    try:
        llm = get_llm_client()
        response = llm.invoke("Say 'GenAI Lab connection successful!' in one line.")
        print(f"✅ SUCCESS: {response.content}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
