import os
from langchain_huggingface import HuggingFaceEndpoint

token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

models = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

for model in models:
    try:
        print(f"\nTesting {model}...")
        llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            temperature=0.2,
            max_new_tokens=100,
            huggingfacehub_api_token=token,
        )
        result = llm.invoke("What is Python?")
        print(f"  SUCCESS: {result[:80]}...")
    except Exception as e:
        print(f"  FAILED: {e}")