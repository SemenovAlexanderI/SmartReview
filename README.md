
---
# SmartReview API

**SmartReview** - это ML/NLP сервис для глубокого анализа пользовательских отзывов. Проект комбинирует классические NLP-задачи (анализ тональности, классификация) с продвинутой системой RAG (Retrieval-Augmented Generation) для получения осмысленных ответов на вопросы о данных.

Сервис построен на **FastAPI**, **Transformers** и **LangChain** (с **Qdrant** в качестве векторной базы) и полностью готов к запуску в **Docker**.

---
##  Core-возможности

* **Классический NLP-анализ (Endpoint: `/predict`)**
    * **Анализ тональности**: Определение (positive/negative) с оценкой уверенности.
    * **Классификация по типу**: Определение категории отзыва (жалоба, похвала, вопрос, предложение) на основе правил.
    * **Суммаризация**: Краткое изложение длинных отзывов с использованием `flan-t5`.
* **Пакетная обработка (Endpoint: `/predict_batch`)**
    * Высокопроизводительная обработка списка отзывов (до 600 за раз), идеально подходит для анализа небольших датасетов.
* **Интеллектуальный RAG-агент (Endpoint: `/ask`)**
    * Позволяет задавать вопросы к базе отзывов (например, к базе данных отелей или продуктов).
    * Использует **семантический роутер** на базе `gpt-4o-mini` для выбора одного из 4 сценариев ответа:
        1.  **Поиск жалоб (`complaints`):** Поиск конкретных технических проблем и недостатков в негативных отзывах.
        2.  **Анализ причин (`root_cause`):** Глубинный анализ *причин* недовольства (в роли Product Manager).
        3.  **Сравнение (`comparison`):** Создание таблицы "Плюсы vs Минусы" на основе позитивных и негативных отзывов.
        4.  **Общий вопрос (`general`):** Ответ на общие вопросы, не требующие поиска по базе.
* **Production-Ready**
    * **Мониторинг**: Готовый эндпоинт `/metrics` для **Prometheus** (отслеживание RPS, задержек, ошибок, использования VRAM).
    * **Rate Limiting**: Встроенный middleware для ограничения частоты запросов по IP.
    * **Логгирование**: Структурированное JSON-логирование для удобного парсинга.
    * **Контейнеризация**: Полная поддержка Docker и Docker Compose (включая GPU-версию).

---
## Архитектура и стек

!скриншот архитектуры 

* **API**: FastAPI
* **NLP-пайплайн**: Transformers (Hugging Face), NLTK (для токенизации предложений)
* **RAG-система**: LangChain
* **LLM**: OpenAI `gpt-4o-mini` (для роутинга и генерации)
* **Embeddings**: OpenAI `text-embedding-3-small`
* **Векторная база**: Qdrant (Cloud)
* **Мониторинг**: Prometheus
* **Оркестрация**: Docker, Docker Compose

---
## Описание ключевых компонентов

### Приложение (`app/`)
- **`main.py`** - FastAPI-приложение: эндпоинты, middleware, логи, метрики
- **`nlp_inference.py`** - Логика NLP: загрузка моделей, sentiment, summarize, batch
- **`rag_service.py`** - Логика RAG: LangChain, Qdrant, роутер и 4 сценария

### Тестирование (`tests/`)
- **`conftest.py`**
- **`test_api.py`** -  API-тесты

###  Исследования (`notebooks/`)
- **`nlp_model_research.ipyn`** - Ноутбуки для R&D NLP-моделей
-  **`rag_testing.ipynb`** - Эксперименты с RAG-агентом

###  Дополнительные компоненты
- **`model_training/`** - Pipeline обучения ML-моделей
- **`Dockerfile.gpu.py312`** - Docker-конфигурации
- **`entrypoint.sh`** - Docker-?????
- **`docker-compose.yml`** - Docker-?????
- Файлы зависимостей ?

### Требования

- Python 3.12+
- Docker (для контейнеризации)
- GPU (опционально, для ускорения inference; поддерживается CUDA)
- API-ключи:
    - OPENAI_API_KEY (для RAG и embeddings).
    - QDRANT_URL и QDRANT_API_KEY (для векторной БД).
    - LANGSMITH_API_KEY (для langsmith).

Библиотеки: Указаны в requirements.txt (FastAPI, Transformers, LangChain, Qdrant, Prometheus, etc.).

---
## Быстрый старт (Docker)

### 1. Предварительные требования

* [Docker](https://www.docker.com/get-started)
* [Docker Compose](https://docs.docker.com/compose/install/)
* (Опционально) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) для поддержки GPU.

### 2. Конфигурация

Для работы RAG-агента и NLP-моделей необходимы переменные окружения.

Создайте файл `.env` в корне проекта:

```bash
# Скопируйте пример
cp .env.example .env
```
И заполните .env своими ключами:

API_KEY нужно получить в своем аккаунте
```
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<***>
OPENAI_API_KEY=<***>
TAVILY_API_KEY=<***>
COLLECTION_NAME=amazon_complaints
LANGSMITH_PROJECT="AMAZON product analysis"
QDRANT_URL=<***>
QDRANT_API_KEY=<***>
LANGCHAIN_TRACING_V2=true 

```

### Запуск 

#### Сборка и запуск через docker‑compose (GPU‑вариант):
************

- Использует Dockerfile.gpu.py312 для GPU-поддержки (если доступно).
`code`

- запуск (если хотите использовать только CPU то )
`code`

- Entry point: entrypoint.sh (инициализирует модели и запускает FastAPI).
- Игнорируемые файлы: В .dockerignore.


API будет на http://localhost:8000. Для мониторинга метрик: http://localhost:8000/metrics.

---
## Использование
в качестве тестового прогона я использовал `postman`, но можно обойтись встроенными инструментами (`http://localhost:8000/docs`)

*** добавить скриншоты как работает в постмане 

### Инициализация моделей

- NLP-модели (sentiment и summarizer) загружаются при старте (в lifespan).
- RAG-агент инициализируется, если есть ключи OpenAI/Qdrant.
- Если NLTK не найден, скачивается автоматически.

*** показать скриншот что модели подгружаются при запуске контейнера , а не при предсказание уже 


### Эндпоинты

- **GET /health:** Проверка статуса (включая RAG).
    - Ответ: {"status": "ok", "rag_enabled": true}

- **POST /predict:** Анализ одиночного отзыва.
    - Тело: {"text": "Отличный отель!", "summary": true}
    - Ответ: {"sentiment": "positive", "sentiment_score": 0.99, "category": "praise", "summary": "Краткое описание..."}

- **POST /predict_batch:** Батч-анализ (до 600 текстов).
    - Тело: {"texts": ["Плохой сервис", "Отличный вид"], "summary": false}
    - Ответ: {"results": [{"sentiment": "negative", "sentiment_score": 0.88, "category": "complaint"}, ...]}

- **POST /ask**: RAG-запрос.
    - Тело: {"question": "Почему гости жалуются на чистоту?"}
    - Ответ: {"question": "...", "answer": "Анализ на основе отзывов..."}
RAG роутит вопрос в один из сценариев (жалобы, причины, сравнение, общий).

#### Примеры запросов (cURL)
- `####`
- `####`
мейби скрины добавить 
![Image 1](https://github.com/{username}/{repository}/raw/{branch}/{path}/image.png)

---
## Мониторинг и логи

- Логи: JSON-формат (stdout), уровень INFO. Включают latency, ошибки, extra-данные.
- Метрики: Prometheus на /metrics (requests, errors, latency, GPU usage, summarizer failures).
- Rate limit: 10 req/min на IP, с Retry-After.

---
## Тестирование
Запустите тесты:
```
pytest tests/
```

Моки для NLP/RAG/Prometheus (чтобы тесты работали без реальных моделей/API).
Покрытие: Health, predict, batch, RAG, ошибки (empty input, limits).