import os
import wikipedia
from dotenv import load_dotenv
from serpapi import GoogleSearch
import re

# Load .env variables (like SERPAPI_KEY)
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def clean_query(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return " ".join(text.split()[:6])

def retrieve_from_wikipedia(query: str, num_sentences: int = 3) -> list[str]:
    try:
        query = clean_query(query)
        page = wikipedia.page(query)
        content = page.content
        sentences = [s.strip() for s in content.split(". ") if s.strip()]
        return sentences[:num_sentences]
    except Exception as e:
        print(f"⚠️ Wikipedia failed for '{query}': {e}")
        return []


def retrieve_from_google(query: str, num_results: int = 3) -> list[str]:
    if not SERPAPI_KEY:
        raise ValueError("Missing SERPAPI_KEY in .env")

    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "hl": "en"
        })

        results = search.get_dict()
        snippets = []
        for result in results.get("organic_results", []):
            snippet = result.get("snippet")
            if snippet:
                snippets.append(snippet)

        return snippets
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return []


def retrieve_evidence(query: str, fallback_to_google: bool = True) -> list[str]:
    print(f"🔎 Trying Wikipedia for: {query}")
    wiki_results = retrieve_from_wikipedia(query)

    if wiki_results:
        print(f"✅ Wikipedia success: {len(wiki_results)} sentences found")
        return wiki_results
    elif fallback_to_google:
        print(f"🔁 Falling back to Google search for: {query}")
        return retrieve_from_google(query)
    else:
        return []
