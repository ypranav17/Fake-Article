import spacy

# Load English NER model
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text: str):
    """
    Extracts named entities from text using spaCy.
    Returns a list of (entity_text, entity_label).
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
