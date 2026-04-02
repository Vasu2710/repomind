import os
import requests

token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

models = [
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

headers = {"Authorization": f"Bearer {token}"}

for model in models:
    try:
        print(f"\nTesting {model}...")
        # Use the chat completions endpoint (conversational)
        response = requests.post(
            f"https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is Python? Reply in one sentence."}],
                "max_tokens": 100,
            },
        )
        data = response.json()
        if "choices" in data:
            print(f"  SUCCESS: {data['choices'][0]['message']['content'][:80]}...")
        else:
            print(f"  FAILED: {data}")
    except Exception as e:
        print(f"  FAILED: {e}")