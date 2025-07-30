import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CLAIMBUSTER_API_KEY")
CLAIMBUSTER_ENDPOINT = "https://idir.uta.edu/claimbuster/api/v2/score/text/"

def get_claimbuster_score(sentence: str) -> float:
    headers = {"x-api-key": API_KEY}
    payload = {"input_text": sentence}

    response = requests.post(CLAIMBUSTER_ENDPOINT, headers=headers, json=payload)

    print(f"🔁 Sent: {sentence}")
    print(f"📥 Status: {response.status_code}")
    print(f"📦 Raw Response: {response.text}")

    if response.status_code == 200:
        result = response.json()
        try:
            return result["results"][0]["score"]
        except (KeyError, IndexError):
            print("⚠️ Could not extract score from result.")
            return 0.0
    else:
        print(f"❌ ClaimBuster API Error: {response.status_code}")
        return 0.0
