# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# --- GLOBAL API METRICS ---
REQUEST_COUNTER = Counter("smartreview_requests_total", "Total API requests", ["endpoint", "method", "status"])
REQUEST_ERRORS = Counter("smartreview_errors_total", "Total API errors", ["endpoint", "method", "exception"])
LATENCY_HIST = Histogram("smartreview_request_latency_seconds", "Request latency seconds", ["endpoint"])
SUMMARIZER_FAILURES = Counter("smartreview_summarizer_failures_total", "Summarizer failures", ["endpoint"])

# --- GPU METRICS ---
INFER_GPU_USED = Gauge("smartreview_gpu_memory_bytes", "GPU memory allocated (bytes)", ["device"])
INFER_GPU_RESERVED = Gauge("smartreview_gpu_reserved_bytes", "GPU reserved bytes", ["device"])
INFER_GPU_MAX_RESERVED = Gauge("smartreview_gpu_max_reserved_bytes", "GPU max reserved bytes", ["device"])

# --- RAG SPECIFIC METRICS ---
RAG_REQUESTS = Counter("smartreview_rag_requests_total", "Total RAG requests processed", ["endpoint", "method", "status"])
RAG_ERRORS = Counter("smartreview_rag_errors_total", "Total RAG errors", ["endpoint", "method", "error_type"]) 
RAG_LATENCY_HIST = Histogram("smartreview_rag_latency_seconds", "RAG processing latency", ["endpoint"])