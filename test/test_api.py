# test/test_api.py
from fastapi.testclient import TestClient
import app.main as main_app

# def test_health():
#     client = TestClient(main_app.app)
#     r = client.get("/health")
#     assert r.status_code == 200
#     assert r.json() == {"status": "ok"}

# def test_predict_single():
#     client = TestClient(main_app.app)
#     payload = {"text": "great stay", "summary": True}
#     r = client.post("/predict", json=payload)
#     assert r.status_code == 200
#     j = r.json()
#     assert j["sentiment"] == "positive"
#     assert j["summary"] == "short"

# def test_predict_batch():
#     client = TestClient(main_app.app)
#     payload = {"texts": ["a","b","c"], "summary": False}
#     r = client.post("/predict_batch", json=payload)
#     assert r.status_code == 200
#     data = r.json()
#     assert "results" in data
#     assert len(data["results"]) == 3

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "rag_enabled": True}

def test_predict_sentiment(client):
    payload = {"text": "This app is amazing!", "summary": True}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["summary"] == "Mock summary"

def test_predict_empty(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_batch(client):
    payload = {"texts": ["bad service", "slow app"], "summary": False}
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["sentiment"] == "negative" # Из нашего мока

def test_rag_ask(client):
    payload = {"question": "Why is the delivery slow?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Mock RAG answer" in data["answer"]