import os
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM
from typing import List, Dict
import time
import nltk
from nltk.tokenize import sent_tokenize
import re


# глобальные переменные для моделей
tokenizer = None
sent_model = None
summarizer = None
NLTK_OK = False
MODEL_INITED = False

# config
NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/opt/conda/nltk_data")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "distilbert-base-uncased")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "google/flan-t5-small")
SENT_MODEL_DIR = os.environ.get("MODEL_DIR", None)
SUMMARY_DEVICE = int(os.environ.get("SUMMARY_DEVICE", "-1"))  # -1 means CPU


# загрузка NLTK punkt ресурса
def ensure_nltk_punkt():
    nltk.data.path.append(NLTK_DATA_DIR)
    candidates = [
        "tokenizers/punkt",
        "tokenizers/punkt/english",
        "tokenizers/punkt_tab/english"  # try to cover the unusual name
    ]
    for res in candidates:
        try:
            nltk.data.find(res)
            print(f"[nlp_inference] Found NLTK resource: {res}")
            return True
        except LookupError:
            continue
    # not found -> try download 'punkt'
    try:
        print("[nlp_inference] NLTK punkt not found, downloading 'punkt' ...")
        nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)
        # try again
        nltk.data.find("tokenizers/punkt")
        print("[nlp_inference] punkt downloaded successfully.")
        return True
    except Exception as e:
        print("[nlp_inference] Warning: failed to download punkt:", e)
    # If still not found, return False (we will fallback later in code)
    return False

# Initialize models
def init_models(force_reload: bool = False) -> dict:
    global tokenizer, sent_model, summarizer, NLTK_OK, MODEL_INITED, SENT_MODEL_DIR

    if MODEL_INITED and not force_reload:
        return {"ok": True, "msg": "already initialized"}
    
    info = {}
    start = time.time()

    # NLTK
    try:
        NLTK_OK = ensure_nltk_punkt()
        info['nltk_ok'] = NLTK_OK
    except Exception as e:
        NLTK_OK = False
        info['nltk_error'] = str(e)

    # tokenizer
    model_path = SENT_MODEL_DIR if (SENT_MODEL_DIR and os.path.exists(SENT_MODEL_DIR)) else None
    try:
        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
            info['tokenizer'] = f"local:{model_path}"
        else:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, use_fast=True)
            info['tokenizer'] = f"hub:{FALLBACK_MODEL}"
    except Exception as e:
        # fallback to hub
        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, use_fast=True)
            info['tokenizer_fallback'] = f"hub:{FALLBACK_MODEL}"
        except Exception as e2:
            info['tokenizer_error'] = str(e2)
            raise

     # загрузка модели 
    try:
        if model_path:
            sent_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            info['sent_model'] = f"local:{model_path}"
        else:
            sent_model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
            info['sent_model'] = f"hub:{FALLBACK_MODEL}"
    except Exception as e:
        # fallback to hub
        try:
            sent_model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
            info['sent_model_fallback'] = f"hub:{FALLBACK_MODEL}"
        except Exception as e2:
            info['sent_model_error'] = str(e2)
            raise

    # перемещенее модели на GPU, если доступно
    if torch.cuda.is_available():
        try:
            sent_model.to(torch.device("cuda"))
            info['sent_model_device'] = "cuda"
        except Exception as e:
            info['sent_model_device_error'] = str(e)
    else:
        info['sent_model_device'] = "cpu"

    info['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(torch.cuda.current_device())

    # summarizer init (may be large)
    try:
        summarizer = pipeline("summarization", model=SUMMARY_MODEL, device=SUMMARY_DEVICE)
        info['summarizer'] = f"{SUMMARY_MODEL} device={SUMMARY_DEVICE}"
    except Exception as e:
        info['summarizer_error_init'] = str(e)
        # try CPU fallback
        try:
            summarizer = pipeline("summarization", model=SUMMARY_MODEL, device=-1)
            info['summarizer_fallback'] = f"{SUMMARY_MODEL} device=-1"
        except Exception as e2:
            info['summarizer_error'] = str(e2)
            summarizer = None

    MODEL_INITED = True
    info['init_time_s'] = round(time.time() - start, 2)
    print("[nlp_inference] init_models:", info)
    return info


# предсказание сентимента для одного текста
def predict_sentiment(text: str) -> Dict:
    sent_model.eval()
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, max_length=512, padding="longest", return_tensors="pt")
        if torch.cuda.is_available():
             enc = {k: v.cuda() for k,v in enc.items()}
        out = sent_model(**enc)
        logits = out.logits
        pred = torch.argmax(logits, dim=-1).cpu().item()
        label = "positive" if pred == 1 else "negative"
        scores = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
        score = max(scores)
    return {"sentiment": label, "score": score}


# жалоба 
COMPLAINT_KW = ['not', 'no', 'dirty', 'broken', 'complain', 'problem', 'issue', 'refund', 'late', 'rude', 'terrible']
# комплименты
PRAISE_KW = ['great', 'excellent', 'clean', 'friendly', 'love', 'amazing', 'perfect', 'good', 'recommend']
# вопрос
QUESTION_KW = ['how', 'what', 'where', 'when', 'why', 'is there', 'do you', 'could you']
# предложение 
SUGGESTION_KW = ['should', 'could', 'would be', 'suggest', 'recommendation', 'idea']

# эвристическая классификация по типу отзыва
def classify_type_rule(text: str) -> str:
    t = text.lower()
    for kw in COMPLAINT_KW:
        if kw in t:
            return 'complaint'
    for kw in PRAISE_KW:
        if kw in t:
            return 'praise'
    for kw in QUESTION_KW:
        if kw in t:
            return 'question'
    for kw in SUGGESTION_KW:
        if kw in t:
            return 'suggestion'
    return 'other'

#chunking + summarization 
def chunk_sentences(text: str, max_chars:int = 800) -> List[str]:
    # усли nltk punkt доступен, используем sent_tokenize; иначе наивный разбор
    sents = None
    if NLTK_OK:
        try:
            sents = sent_tokenize(text)
        except Exception as e:
            print("[nlp_inference] Warning: sent_tokenize failed:", e)
            sents = None
    if not sents:
        # naive fallback: split on punctuation + keep reasonably sized segments
        parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not parts:
            parts = [text]
        sents = parts

    chunks = []
    curr = []
    curr_len = 0
    for s in sents:
        if curr_len + len(s) > max_chars and curr:
            chunks.append(" ".join(curr))
            curr = [s]
            curr_len = len(s)
        else:
            curr.append(s)
            curr_len += len(s)
    if curr:
        chunks.append(" ".join(curr))
    return chunks

import logger
# безопасная суммаризация с обработкой ошибок
def summarize_chunk_safe(text_chunk: str, max_length=50, min_length=8):
    global summarizer
    if summarizer is None:
        logger.warning("[nlp_inference] summarizer not initialized")
        raise RuntimeError("summarizer_not_init")
    try:
        out = summarizer(text_chunk, max_new_tokens=128, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("summary_text") or out[0].get("generated_text")
        return None
    except Exception as e:
        # метка: это summarizer ошибка
        e.is_summarizer_error = True
        logger.error("[nlp_inference] summarizer failed: %s", e)
        raise

# суммаризация длинного текста с разбиением на чанки
def summarize_long_text(text: str, max_chars=800) -> str:
    chunks = chunk_sentences(text, max_chars=max_chars)
    parts = []
    for c in chunks:
        s = summarize_chunk_safe(c, max_length=50, min_length=8)
        if s:
            parts.append(s)
        else:
            print("[nlp_inference] Warning: skipping chunk due summarizer failure")
    if not parts:
        return None
    combined = " ".join(parts)
    if len(combined) > 400:
        final = summarize_chunk_safe(combined, max_length=80, min_length=20)
        return final
    return combined

# анализ текста: сентимент, тип, (опционально) саммари
def analyze_text(text: str, do_summary: bool = True) -> Dict:
    sent = predict_sentiment(text)
    typ = classify_type_rule(text)   
    summary = None
    if do_summary:
        summary = summarize_long_text(text, max_chars=800)
    return {
        "sentiment": sent["sentiment"],
        "sentiment_score": sent["score"],
        "category": typ,
        "summary": summary
    }

from torch.utils.data import DataLoader, TensorDataset

# batch inferes
def batch_predict_sentiment(texts: List[str], batch_size: int = 32):
    sent_model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k,v in enc.items()}
        with torch.no_grad():
            out = sent_model(**enc)
            logits = out.logits.cpu()
            preds = torch.argmax(logits, dim=-1).numpy().tolist()
            probs = torch.softmax(logits, dim=-1).numpy().tolist()
        for p, prob in zip(preds, probs):
            label = "positive" if p == 1 else "negative"
            score = max(prob)
            results.append({"sentiment": label, "score": float(score)})
    return results