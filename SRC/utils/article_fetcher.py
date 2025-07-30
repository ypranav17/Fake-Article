from newspaper import Article
import trafilatura
import requests
from bs4 import BeautifulSoup

def extract_article_from_url(url: str) -> str:
    try:
        # ✅ Try newspaper3k first
        article = Article(url)
        article.download()
        article.parse()
        full_text = article.text.strip()
        if full_text and len(full_text.split()) > 50:
            print("✅ Extracted FULL article using Newspaper3k")
            return full_text
        else:
            print("⚠️ Newspaper3k text too short, falling back...")
    except Exception as e:
        print(f"⚠️ Newspaper3k failed: {e} — falling back to Trafilatura.")

    try:
        # 🟡 Try trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.split()) > 50:
                print("✅ Extracted using Trafilatura")
                return text.strip()
            else:
                print("⚠️ Trafilatura text too short, falling back...")
    except Exception as e:
        print(f"⚠️ Trafilatura failed: {e} — falling back to raw scraping.")

    try:
        # 🔴 Final fallback: basic scraping with BeautifulSoup
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs).strip()
        if len(text.split()) > 50:
            print("✅ Extracted using BeautifulSoup fallback")
            return text
        else:
            print("⚠️ BeautifulSoup text too short.")
    except Exception as e:
        print(f"❌ Final fallback failed: {e}")

    return "⚠️ Failed to extract article text."
