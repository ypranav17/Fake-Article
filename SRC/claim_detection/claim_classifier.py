from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import torch.nn.functional as F
from evidence.retrieval import retrieve_evidence
from claim_detection.claimbuster_client import get_claimbuster_score
from utils.article_fetcher import extract_article_from_url

# ✅ Switched model for manual mode
MANUAL_MODEL_ID = "mrm8488/bert-tiny-finetuned-fake-news-detection"
AUTO_MODEL_ID = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

# Load both models
manual_tokenizer = AutoTokenizer.from_pretrained(MANUAL_MODEL_ID)
manual_model = AutoModelForSequenceClassification.from_pretrained(MANUAL_MODEL_ID)

auto_tokenizer = AutoTokenizer.from_pretrained(AUTO_MODEL_ID, use_fast=False)
auto_model = AutoModelForSequenceClassification.from_pretrained(AUTO_MODEL_ID)

CONFIDENCE_THRESHOLD = 0.92
WEAK_ENTAILMENT_THRESHOLD = 0.85

# Google Programmable Search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# Wikipedia entity resolver
def fetch_wikipedia_summary(title):
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={title}&format=json"
        search_resp = requests.get(search_url).json()
        results = search_resp.get("query", {}).get("search", [])
        if not results:
            return None
        top_title = results[0]['title']
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_title.replace(' ', '_')}"
        summary_resp = requests.get(summary_url).json()
        return summary_resp.get("extract", "").lower()
    except Exception as e:
        print(f"❌ Wikipedia error: {e}")
        return None

def dynamic_wiki_check(text):
    entities = extract_named_entities(text)
    text_lower = text.lower()

    contradiction_pairs = [
        ("narendra modi", "president of the united states", "joe biden"),
        ("joe biden", "prime minister of india", "narendra modi"),
        ("elon musk", "president of the united states", "joe biden"),
        ("barack obama", "current president", "joe biden"),
    ]

    for ent in entities:
        snippet = fetch_wikipedia_summary(ent)
        if not snippet:
            continue

        # Contradiction detection logic
        for entity, role, correct_person in contradiction_pairs:
            if entity in text_lower and role in text_lower and correct_person not in snippet:
                print(f"❌ Contradiction: '{entity}' assigned wrong role '{role}'")
                return True

        # Extra catch: if entity isn't mentioned in its own snippet, it's suspicious
        if ent.lower() not in snippet:
            print(f"⚠️ Entity '{ent}' not in its own summary → suspicious")
            return True

    return False



def google_search(query):
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
        resp = requests.get(url)
        data = resp.json()
        return [item['snippet'] for item in data.get('items', [])]
    except:
        return []

def classify_nli(text: str, mode: str = "auto") -> dict:
    premise = "According to verified sources,"
    tokenizer = auto_tokenizer if mode == "auto" else manual_tokenizer
    model = auto_model if mode == "auto" else manual_model

    inputs = tokenizer(premise, text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    label_id = torch.argmax(probs).item()
    label_name = id2label[label_id]
    score = probs[label_id].item()

    final_label = "UNSURE"
    if score >= CONFIDENCE_THRESHOLD:
        if label_name == "ENTAILMENT":
            final_label = "REAL"
        elif label_name == "CONTRADICTION":
            final_label = "FAKE"

    return {
        "nli_label": label_name,
        "score": score,
        "probs": probs.tolist(),
        "label": final_label
    }

def is_fallback_fake(evidence_list):
    suspicious = ["conspiracy", "disproven", "misinformation", "false", "debunked", "not true", "fake"]
    return any(any(word in e.lower() for word in suspicious) for e in evidence_list)

def classify_manual_text(text: str) -> str:
    if not text.strip():
        return "UNSURE"

    result = classify_nli(text, mode="manual")
    print(f"🧾 Manual Claim: {text}")
    print(f"➡️ Label: {result['label']} ({result['nli_label']}), Score: {result['score']:.2f}")

    needs_extra_check = (
        result["label"] == "UNSURE" or
        result["nli_label"] == "ENTAILMENT" and result["score"] < WEAK_ENTAILMENT_THRESHOLD or
        dynamic_wiki_check(text)
    )

    if needs_extra_check:
        google_evidence = google_search(text)
        print(f"🔎 Google Search Evidence: {google_evidence}")
        if not google_evidence:
            fallback = retrieve_evidence(text)
            print(f"🔁 Fallback Evidence: {fallback}")
            if not fallback:
                return "UNSURE"
            return "FAKE" if is_fallback_fake(fallback) else "REAL"
        return "FAKE" if is_fallback_fake(google_evidence) else "REAL"

    return result["label"]

def classify_claim_auto(text: str, mode: str = "article") -> str:
    if not text.strip():
        return "UNSURE"

    cb_score = get_claimbuster_score(text)
    print(f"📊 ClaimBuster score: {cb_score:.2f}")

    result = classify_nli(text, mode="auto")
    print(f"📰 Article Claim: {text}")
    print(f"➡️ Label: {result['label']} ({result['nli_label']}), Score: {result['score']:.2f}")

    if mode == "article":
        retrieve_evidence(text)

    needs_extra_check = (
        result["label"] == "UNSURE" or
        result["nli_label"] == "ENTAILMENT" and result["score"] < WEAK_ENTAILMENT_THRESHOLD or
        dynamic_wiki_check(text)
    )

    if needs_extra_check:
        fallback = retrieve_evidence(text)
        print(f"🔁 Fallback Evidence: {fallback}")
        if not fallback:
            return "UNSURE"
        return "FAKE" if is_fallback_fake(fallback) else "REAL"
    return result["label"]