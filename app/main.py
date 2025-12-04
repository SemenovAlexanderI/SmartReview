# app/main.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from app import nlp_inference
from app import rag_service 
from starlette.responses import PlainTextResponse
import time
from collections import defaultdict
from contextlib import asynccontextmanager
import asyncio
import logging
import traceback
import torch
import gc
import json, sys

# --- logger (declare early, чтобы использовать в teardown и других местах) ---
logger = logging.getLogger("smartreview")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    class JsonFormatter(logging.Formatter):
        # кастомизируем формат логов в JSON
        def format(self, record): # кастомизи
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage(),
                "name": record.name,
            }
            # приклеим extra
            # Добавляет в payload все extra-поля из record
            for k, v in record.__dict__.items():
                if k not in ("args","asctime","created","exc_info","exc_text","filename","funcName",
                             "levelname","levelno","lineno","module","msecs","msg","name","pathname",
                             "process","processName","relativeCreated","stack_info","thread","threadName"):
                    payload[k] = v
            return json.dumps(payload, ensure_ascii=False)
    # Прикрепляет formatter к handler.
    h.setFormatter(JsonFormatter())

    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# lifespan: современный get_running_loop и аккуратный teardown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # инициализация nlp модуля
    loop = asyncio.get_running_loop() # цикл событий
    try:
        #
        logger.info("Initializing NLP models")
        await loop.run_in_executor(None, nlp_inference.init_models)
        logger.info("models initialized")
        print(f"models initialized")
    except Exception as e:
        logger.exception("init_models during lifespan failed: %s", e)
        raise

    # инициализация rag agent
    try:
        logger.info("Initializing RAG agent")
        print(f"Initializing RAG agent")
        await loop.run_in_executor(None, rag_service.init_rag_agent)
    except Exception as e:
        logger.error(f"RAG init failed: {e}")


    try:
        yield
    finally:
        #cleanup
        try:
            for attr in ("sent_model", "summarizer"):
                if hasattr(nlp_inference, attr):
                    setattr(nlp_inference, attr, None)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    logger.exception("Error while emptying cuda cache")
            gc.collect()
            logger.info("lifespan teardown finished: cleared models and caches")
        except Exception as e:
            logger.warning("Error during shutdown cleanup: %s", e)


# Create app (после logger и lifespan)
app = FastAPI(lifespan=lifespan, title='SmartReview Insight')

# --- Middleware rate limit ---

RATE_LIMIT = 10
RATE_PERIOD = 60.0
MAX_BODY_BYTES = 512 * 1024  # чуть больше для batch
buckets = defaultdict(lambda: {"tokens": RATE_LIMIT, "last": time.time(), "lock": asyncio.Lock()})

@app.middleware("http")
async def guard_and_limit(request: Request, call_next):
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")

    ip = "unknown"
    try:
        ip = request.client.host if request.client and request.client.host else "unknown"
    except Exception:
        pass

    bucket = buckets[ip]
    async with bucket["lock"]:
        now = time.time()
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(RATE_LIMIT, bucket["tokens"] + elapsed * (RATE_LIMIT / RATE_PERIOD))
        bucket["last"] = now
        if bucket["tokens"] < 1:
            # Доп. подсказки клиенту
            headers = {"Retry-After": str(int(RATE_PERIOD))}
            raise HTTPException(status_code=429, detail="Too many requests", headers=headers)
        bucket["tokens"] -= 1

    response = await call_next(request)
    return response

# --- pydantic models ---

class PredictRequest(BaseModel):
    text: str
    summary: bool = True

class PredictBatchRequest(BaseModel):
    texts: List[str]
    summary: bool = False

class RagRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "rag_enabled": rag_service.rag_enabled}

# --- METRICS ENDPOINT ---
from app.metrics import (
    REQUEST_COUNTER, REQUEST_ERRORS, LATENCY_HIST, SUMMARIZER_FAILURES,
    INFER_GPU_USED, INFER_GPU_RESERVED, INFER_GPU_MAX_RESERVED,
    RAG_REQUESTS, RAG_ERRORS, RAG_LATENCY_HIST # <-- обновили импорт
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

## ---- NLP ENDPOINTS (predict and predict batch) ----
# --- pipeline predict (single) ---
@app.post("/predict")
def predict(payload: PredictRequest):
    if not payload.text or not payload.text.strip():
        REQUEST_COUNTER.labels("/predict", "POST", "422").inc()
        raise HTTPException(status_code=422, detail="Text is empty")

    t0 = time.time()
    try:
        out = nlp_inference.analyze_text(payload.text, do_summary=payload.summary)
        latency = time.time() - t0

        REQUEST_COUNTER.labels("/predict", "POST", "200").inc()
        LATENCY_HIST.labels("/predict").observe(latency)

        # GPU memory metrics
        if torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
            try:
                INFER_GPU_USED.labels(dev).set(torch.cuda.memory_allocated())
                INFER_GPU_RESERVED.labels(dev).set(torch.cuda.memory_reserved())
                INFER_GPU_MAX_RESERVED.labels(dev).set(torch.cuda.max_memory_reserved())
            except Exception:
                logger.exception("Error reading CUDA memory stats")

        logger.info("predict finished", extra={"latency_s": latency, "text_len": len(payload.text)})
        return out

    except Exception as e:
        latency = time.time() - t0
        REQUEST_ERRORS.labels("/predict", "POST", e.__class__.__name__).inc()
        logger.error("predict error: %s\n%s", e, traceback.format_exc())
        if getattr(e, "is_summarizer_error", False):
            SUMMARIZER_FAILURES.labels("/predict").inc()
        raise HTTPException(status_code=500, detail="Internal server error")


# --- batch predict ---
MAX_TEXTS_PER_REQUEST = 600

@app.post("/predict_batch")
def predict_batch(payload: PredictBatchRequest):
    texts = payload.texts or []
    if len(texts) > MAX_TEXTS_PER_REQUEST:
        REQUEST_COUNTER.labels("/predict_batch", "POST", "413").inc()
        raise HTTPException(status_code=413, detail=f"Too many texts; max {MAX_TEXTS_PER_REQUEST}")

    if not texts:
        REQUEST_COUNTER.labels("/predict_batch", "POST", "422").inc()
        raise HTTPException(status_code=422, detail="No texts provided")

    t0 = time.time()
    try:
        sents = nlp_inference.batch_predict_sentiment(texts, batch_size=32)
        res = []
        for i, t in enumerate(texts):
            item = {"sentiment": sents[i]["sentiment"], "sentiment_score": sents[i]["score"]}
            item["category"] = nlp_inference.classify_type_rule(t)
            if payload.summary:
                item["summary"] = nlp_inference.summarize_long_text(t)
            res.append(item)

        latency = time.time() - t0
        REQUEST_COUNTER.labels("/predict_batch", "POST", "200").inc()
        LATENCY_HIST.labels("/predict_batch").observe(latency)

        if torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
            try:
                INFER_GPU_USED.labels(dev).set(torch.cuda.memory_allocated())
                INFER_GPU_RESERVED.labels(dev).set(torch.cuda.memory_reserved())
                INFER_GPU_MAX_RESERVED.labels(dev).set(torch.cuda.max_memory_reserved())
            except Exception:
                logger.exception("Error reading CUDA memory stats")

        logger.info("predict_batch finished", extra={"latency_s": latency, "n_texts": len(texts)})
        return {"results": res}

    except Exception as e:
        latency = time.time() - t0
        REQUEST_ERRORS.labels("/predict_batch", "POST", e.__class__.__name__).inc()
        logger.error("predict_batch error: %s\n%s", e, traceback.format_exc())
        if getattr(e, "is_summarizer_error", False):
            SUMMARIZER_FAILURES.labels("/predict_batch").inc()
        raise HTTPException(status_code=500, detail="Internal server error")

# --- RAG ENDPOINT ---
@app.post("/ask")
async def ask_rag(payload: RagRequest):
    """
    Интеллектуальный поиск по базе отзывов с использованием RAG-Router.
    """
    # Валидация входных данных
    if not payload.question or not payload.question.strip():
        RAG_ERRORS.labels("/ask", "POST", "422").inc()
        REQUEST_COUNTER.labels("/ask", "POST", "422").inc() 
        raise HTTPException(status_code=422, detail="Question is empty")

    # Проверка доступности сервиса
    if not rag_service.rag_enabled:
        RAG_ERRORS.labels("/ask", "POST", "503").inc()
        REQUEST_COUNTER.labels("/ask", "POST", "503").inc()
        raise HTTPException(status_code=503, detail="RAG service is unavailable")

    t0 = time.time()
    try:
        # 
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, rag_service.query_rag, payload.question)
        
        latency = time.time() - t0

        RAG_REQUESTS.labels("/ask", "POST", "200").inc()
        REQUEST_COUNTER.labels("/ask", "POST", "200").inc()
        
        RAG_LATENCY_HIST.labels("/ask").observe(latency)
        LATENCY_HIST.labels("/ask").observe(latency)
        
        logger.info("rag request finished", extra={"latency_s": latency, "question": payload.question})
        return {"question": payload.question, "answer": answer}

    except Exception as e:
        latency = time.time() - t0
        error_type = e.__class__.__name__
        
        RAG_ERRORS.labels("/ask", "POST", error_type).inc()
        REQUEST_ERRORS.labels("/ask", "POST", error_type).inc()
        
        logger.error("rag error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal RAG error")