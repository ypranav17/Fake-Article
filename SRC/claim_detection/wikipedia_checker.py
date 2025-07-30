import wikipedia

def check_with_wikipedia(claim: str) -> str:
    try:
        results = wikipedia.search(claim)
        if not results:
            return "UNSURE"
        
        summary = wikipedia.summary(results[0], sentences=2).lower()
        claim_lower = claim.lower()

        print(f"🔍 Wiki Match Summary: {summary}")

        if claim_lower in summary or any(word in summary for word in claim_lower.split()):
            return "REAL"
        else:
            return "UNSURE"
    except Exception as e:
        print(f"⚠️ Wikipedia Error: {e}")
        return "UNSURE"
