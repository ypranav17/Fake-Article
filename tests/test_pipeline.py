from pipeline.full_pipeline import process_article

def test_pipeline():
    url = "https://www.bbc.com/news/world-asia-india-66222263"  # or any short valid article
    results = process_article(url)
    assert isinstance(results, list)
    assert all("sentence" in r for r in results)
    assert all("claimbuster_score" in r for r in results)
