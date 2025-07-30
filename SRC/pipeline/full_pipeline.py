from utils.article_fetcher import extract_article_from_url
from utils.text_preprocessor import split_into_sentences
from ner.ner_pipeline import extract_named_entities
from evidence.retrieval import retrieve_evidence
from claim_detection.claim_classifier import classify_claim_auto
from claim_detection.claimbuster_client import get_claimbuster_score
from claim_detection.claim_classifier import classify_manual_text


def run_pipeline_from_url(url: str):
    mode = "article"
    text = extract_article_from_url(url)
    print(f"\n📄 Extracted article:\n{text}\n")

    raw_sentences = split_into_sentences(text)
    sentences = [s.strip().replace("\n", " ") for s in raw_sentences if s.strip()]
    print(f"\n📝 Sentences extracted ({len(sentences)}):\n{sentences}\n")

    results = []

    for sentence in sentences:
        entities = extract_named_entities(sentence)
        score = get_claimbuster_score(sentence)
        claim_type = classify_claim_auto(sentence)
        evidence = retrieve_evidence(sentence) if score > 0.6 else []

        results.append({
            "sentence": sentence,
            "entities": entities,
            "score": round(score, 2),
            "evidence": evidence,
            "verdict": "Check-worthy" if score > 0.6 else "Not significant",
            "claim_type": claim_type
        })

    return results

def run_pipeline_from_text_manual(text: str):
    sentences = split_into_sentences(text)
    results = []

    for sentence in sentences:
        claim_type = classify_claim_auto(sentence, mode="manual")
        results.append({
            "sentence": sentence,
            "score": 1.0,
            "claim_type": claim_type,
            "evidence": []
        })

    return results

    

