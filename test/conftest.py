# # test/conftest.py
# import sys
# import types
# import pytest

# # --- top-level: вставляем легкие фейки, чтобы import app.main не инициализировал тяжелые зависимости ---
# # fake prometheus_client (минимально)
# _fake_prom = types.ModuleType("prometheus_client")
# _fake_prom.generate_latest = lambda: b""
# _fake_prom.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
# class _DummyMetric:
#     def labels(self, *a, **k): return self
#     def inc(self, *a, **k): pass
#     def set(self, *a, **k): pass
#     def observe(self, *a, **k): pass
# _fake_prom.Counter = lambda *a, **k: _DummyMetric()
# _fake_prom.Histogram = lambda *a, **k: _DummyMetric()
# _fake_prom.Gauge = lambda *a, **k: _DummyMetric()
# sys.modules.setdefault("prometheus_client", _fake_prom)

# # fake minimal nlp_inference module so app.main import doesn't load heavy models during collection
# if "nlp_inference" not in sys.modules:
#     fake_nlp = types.ModuleType("nlp_inference")
#     # minimal stubs used by app.main at import time
#     fake_nlp.init_models = lambda force_reload=False: {"ok": True}
#     fake_nlp.analyze_text = lambda text, do_summary=True: {"sentiment":"positive","sentiment_score":0.9,"category":"praise","summary":("short" if do_summary else None)}
#     fake_nlp.batch_predict_sentiment = lambda texts, batch_size=32: [{"sentiment":"positive","score":0.9} for _ in texts]
#     fake_nlp.classify_type_rule = lambda t: "praise"
#     fake_nlp.summarize_long_text = lambda t: "short"
#     sys.modules["nlp_inference"] = fake_nlp

# # --- fixture: даём возможность тестам перезаписать/подменить поведение при необходимости ---
# @pytest.fixture(autouse=True)
# def make_test_fakes(monkeypatch):
#     """
#     Авто-fixture: если тест хочет замокать что-то другое, он может использовать monkeypatch,
#     но по умолчанию здесь ничего не делаем — у нас уже есть safe defaults.
#     """
#     # если нужно, можно monkeypatch.setattr(...) здесь
#     yield


import sys
import types
import pytest
from unittest.mock import MagicMock
from starlette.testclient import TestClient

# 1. MOCK nlp_inference
# Создаем фейковый модуль, чтобы FastAPI не грузил PyTorch
fake_nlp = types.ModuleType("app.nlp_inference")
fake_nlp.init_models = lambda force_reload=False: {"ok": True}
fake_nlp.analyze_text = lambda text, do_summary=True: {
    "sentiment": "positive",
    "sentiment_score": 0.99,
    "category": "praise",
    "summary": "Mock summary" if do_summary else None
}
fake_nlp.batch_predict_sentiment = lambda texts, batch_size=32: [
    {"sentiment": "negative", "score": 0.88} for _ in texts
]
fake_nlp.classify_type_rule = lambda t: "complaint"
fake_nlp.summarize_long_text = lambda t: "Batch summary"

sys.modules["app.nlp_inference"] = fake_nlp
sys.modules["nlp_inference"] = fake_nlp # На случай абсолютного импорта

# 2. MOCK rag_service
# Нам нужно замокать init и query, чтобы не лезть в OpenAI
fake_rag = types.ModuleType("app.rag_service")
fake_rag.init_rag_agent = lambda: None # Ничего не делаем
fake_rag.rag_enabled = True # Притворяемся, что RAG включен
fake_rag.query_rag = lambda q: f"Mock RAG answer for: {q}"

sys.modules["app.rag_service"] = fake_rag
sys.modules["rag_service"] = fake_rag

# 3. MOCK prometheus (как у тебя было)
fake_prom = types.ModuleType("prometheus_client")
fake_prom.generate_latest = lambda: b"metrics_stub"
fake_prom.CONTENT_TYPE_LATEST = "text/plain"
class _Dummy:
    def labels(self, *a, **k): return self
    def inc(self, *a, **k): pass
    def set(self, *a, **k): pass
    def observe(self, *a, **k): pass
fake_prom.Counter = lambda *a, **k: _Dummy()
fake_prom.Histogram = lambda *a, **k: _Dummy()
fake_prom.Gauge = lambda *a, **k: _Dummy()
sys.modules["prometheus_client"] = fake_prom

# 4. Фикстура клиента
@pytest.fixture
def client():
    # Импортируем app только ПОСЛЕ того, как подменили модули
    from app.main import app
    
    # Можно отключить rate limit для тестов, если мешает, 
    # но middleware уже загружено. Просто шлем запросы медленно или мокаем время.
    
    with TestClient(app) as c:
        yield c