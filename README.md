# SmartReview API

**SmartReview** - это ML/NLP сервис для глубокого анализа пользовательских отзывов. Проект комбинирует классические NLP-задачи (анализ тональности, классификация) с продвинутой системой RAG (Retrieval-Augmented Generation) для получения осмысленных ответов на вопросы о данных.

Сервис построен на **FastAPI**, **Transformers** и **LangChain** (с **Qdrant** в качестве векторной базы) и полностью готов к запуску в **Docker**.

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

## Архитектура и стек
* **API**: FastAPI
* **NLP-пайплайн**: Transformers (Hugging Face), NLTK (для токенизации предложений)
* **RAG-система**: LangChain
* **LLM**: OpenAI `gpt-4o-mini` (для роутинга и генерации)
* **Embeddings**: OpenAI `text-embedding-3-small`
* **Векторная база**: Qdrant (Cloud)
* **Мониторинг**: Prometheus
* **Оркестрация**: Docker, Docker Compose

![Image 1](https://github.com/SemenovAlexanderI/SmartReview/raw/main/Архитектура.png)

## Описание ключевых компонентов

### Приложение (`app/`)
- **`main.py`** - FastAPI-приложение: эндпоинты, middleware, логи, метрики
- **`nlp_inference.py`** - Логика NLP: загрузка моделей, sentiment, summarize, batch
- **`rag_service.py`** - Логика RAG: LangChain, Qdrant, роутер и 4 сценария

### Тестирование (`tests/`)
- **`conftest.py`**
- **`test_api.py`** -  API-тесты

###  Исследования (`notebook_/`)
- **`nlp_model_research.ipynb`** - Ноутбуки для R&D NLP-моделей
-  **`rag_testing.ipynb`** - Эксперименты с RAG-агентом

###  Дополнительные компоненты
- **`train_/`** - Pipeline обучения ML-моделей
- **`Dockerfile.gpu.py312`** - Docker-конфигурации
- **`entrypoint.sh`**
- **`docker-compose.yml`**
- **`reqirements.txt`** 

### Требования
- Python 3.12+
- Docker (для контейнеризации)
- GPU (опционально, для ускорения inference; поддерживается CUDA)
- API-ключи:
    - OPENAI_API_KEY (для RAG и embeddings).
    - QDRANT_URL и QDRANT_API_KEY (для векторной БД).
    - LANGSMITH_API_KEY (для langsmith).

Библиотеки: Указаны в requirements.txt (FastAPI, Transformers, LangChain, Qdrant, Prometheus, etc.).

## Быстрый старт (Docker)
### 1. Предварительные требования
* [Docker](https://www.docker.com/get-started)
* [Docker Compose](https://docs.docker.com/compose/install/)
* (Опционально) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) для поддержки GPU.

### 2. Конфигурация
Для работы RAG-агента и NLP-моделей необходимы переменные окружения.

Создайте файл `.env` в корне проекта:

```
# Скопируйте пример
cp .env.example .env
```
И заполните .env своими ключами:

API_KEY (от openai, langsmith и qdrant) нужно получить в своем аккаунте.
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
***

Для запуска проекта доступны два основных способа: простой (с использованием `docker-compose`) и продвинутый (с ручной сборкой и запуском контейнера). Рекомендуется использовать `docker-compose`.

1. Быстрый запуск с помощью Docker Compose (рекомендуется)
    Если у вас есть GPU и установлены docker и docker-compose, это самый простой способ.

    1) Убедитесь, что ваша модель находится по корректному пути (по умолчанию, 2)указанному в docker-compose.yml).

    2) Создайте файл .env в корне проекта на основе примера (например, .env.example), если он еще не создан, и при необходимости настройте переменные окружения.

    3) Запустите сборку и запуск сервиса одной командой:
    ```
    docker compose up
    ```
    4) После успешного запуска API будет доступно по адресу http://localhost:8000.

    5) Для просмотра метрик перейдите по адресу http://localhost:8000/metrics.

2. Сборка и запуск вручную (GPU-вариант)
Этот способ дает больше контроля над процессом. Используйте его, если вам нужна кастомизация, отсутствующая в docker-compose.

    1) Для начала собирем docker image для этого воспользуемся Dockerfile.gpu.py312 для GPU-поддержки (если доступно).

    `docker build -f Dockerfile.gpu.py312 -t smartreview:py312-gpu .`

    2) Запустите контейнер:

    Запустите собранный образ, указав путь к вашей обученной модели и необходимые переменные окружения:
    ```
    docker run --rm -it --gpus all \
        -p 8000:8000 \
        --env-file .env \
        -v /home/sai/Desktop/smartreview/smartreview/models/sentiment-distilbert_all/checkpoint-337500:/app/models/sentiment:ro \
        -e MODEL_DIR=/app/models/sentiment \
        -e UVICORN_WORKERS=1 \
        -e TOKENIZERS_PARALLELISM=false \
        -e OMP_NUM_THREADS=2 \
        -e SUMMARY_DEVICE=0 \
        --name smartreview_py312_gpu \
        smartreview:py312-gpu
    ```


- Entry point: entrypoint.sh (инициализирует модели и запускает FastAPI - в коде init_models` в lifespan).
- Игнорируемые файлы: В .dockerignore.

API будет на http://localhost:8000. Для мониторинга метрик: http://localhost:8000/metrics.

## Использование
в качестве приложения тестирования я использовал `postman`, но можно обойтись встроенными инструментами (`http://localhost:8000/docs`)

(примеры работы будут в разделе **Эндпоинты**)

### Инициализация моделей

- NLP-модели (sentiment и summarizer) загружаются при старте (в lifespan).
- RAG-агент инициализируется, если есть ключи OpenAI/Qdrant.
- Если NLTK не найден, скачивается автоматически.

![Image start_docker](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/start_docker.png)

### Эндпоинты

- **GET /health:** Проверка статуса (включая RAG).
    - Ответ: {"status": "ok", "rag_enabled": true}

![Image health](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/health.png)

- **POST /predict:** Анализ одиночного отзыва.
    - Тело: {"text": "Отличный отель!", "summary": true}
    - Ответ: {"sentiment": "positive", "sentiment_score": 0.99, "category": "praise", "summary": "Краткое описание..."}

![Image predict](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/predict_.png)

- **POST /predict_batch:** Батч-анализ (до 600 текстов).
    - Тело: {"texts": ["Плохой сервис", "Отличный вид"], "summary": false}
    - Ответ: {"results": [{"sentiment": "negative", "sentiment_score": 0.88, "category": "complaint"}, ...]}

![Image predict_batch](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/predict_batch.png)

- **POST /ask**: RAG-запрос.
    - Тело: {"question": "Почему гости жалуются на чистоту?"}
    - Ответ: {"question": "...", "answer": "Анализ на основе отзывов..."}
RAG роутит вопрос в один из сценариев (жалобы, причины, сравнение, общий).

![Image ask](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/asc(rag).png)

## Мониторинг и логи

- Логи: JSON-формат (stdout), уровень INFO. Включают latency, ошибки, extra-данные.
- Метрики: Prometheus на /metrics (requests, errors, latency, GPU usage, summarizer failures).
- Rate limit: 10 req/min на IP, с Retry-After.
- Трассировка RAG в LangSmith: Для детального анализа работы RAG (LangChain) включена трассировка. Если вы настроите ключи LangSmith в .env (см. раздел Установка), все вызовы chains и роутера будут логироваться в проекте "AMAZON product analysis". В личном кабинете LangSmith вы можете просмотреть трассы: latency по шагам, промпты, ретрив, выводы LLM. Это полезно для отладки и оптимизации (например, увидеть, как роутер выбирает сценарий).

Пример трассы в LangSmith:

![Image langsmith](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/langsmith_1.png)

![Image trace_rag](https://github.com/SemenovAlexanderI/SmartReview/raw/main/PrtSc/trace_rag.png)

## Тестирование
Запустите тесты:
- нужно перейти в окружение проекта и запустить этот код (использовав интерпретатор в котором установлена библиотека pytest)
```
cd test
pytest -v
```

Моки для NLP/RAG/Prometheus (чтобы тесты работали без реальных моделей/API).
Покрытие: Health, predict, batch, RAG, ошибки (empty input, limits).

## Research & Development (R&D)

Разработка проекта началась не с написания бэкенда, а с фазы исследований. Этот этап позволил валидировать гипотезы, выбрать оптимальный стек и отказаться от нерабочих решений до начала продакшен-разработки. 

Все эксперименты задокументированы в Jupyter Notebooks в папке `/notebook_`:

### 1. NLP Exploration & Heuristics (`nlp_module_research.ipynb`)
**Цель:** Понять природу данных и определить границы применимости классических алгоритмов vs LLM.
* **EDA (Exploratory Data Analysis):** Проведен анализ распределения длин текстов, выявлены аномалии и дисбаланс классов.
* **Hypothesis Testing:** Сравнил точность rule-based классификатора (на основе ключевых слов) с ML-подходом. 
    * *Результат:* Эвристики работают быстро, но дают низкую точность. Принято решение использовать Fine-Tuned BERT для классификации.
* **Context Window Optimization:** Для задачи суммаризации длинных отзывов реализован алгоритм **Text Chunking** (нарезка на смысловые блоки), так как "сырые" отзывы часто превышают контекст модели `Flan-T5`.

### 2. RAG Architecture PoC (`rag_testing.ipynb`)
**Цель:** Прототипирование "мозга" системы — агента и поиска.
* **Vector Search Engine:** Настройка Qdrant. Реализован **Payload Index** для ускорения фильтрации по метаданным (фильтр `metadata.sentiment`).
* **Semantic Routing Pattern:** Вместо одной жесткой цепочки (Chain) я реализовал **Router Agent**. 
    * *Логика:* LLM классифицирует интент пользователя (жалоба / вопрос / сравнение) и направляет запрос в специализированный Retriever.
    * *Итог:* Это снизило галлюцинации модели при ответах на специфические вопросы.

## ML Training Pipeline

Для классификации тональности (Sentiment Analysis) был разработан воспроизводимый пайплайн обучения модели `DistilBERT`. Скрипт обучения вынесен из ноутбуков в отдельный модуль `train_.py`.

### Ключевые особенности реализации:
* **Reproducibility:** Жесткая фиксация `SEED` (numpy, torch, python) гарантирует, что результаты экспериментов воспроизводимы.
* **Feature Engineering:** В пайплайн встроена логика автоматической разметки типов обращений (Complaint/Praise/Question) на основе эвристик для обогащения метаданных.
* **Efficiency & Optimization:**
    * Использование **Mixed Precision (FP16)** для ускорения обучения на GPU.
    * **Gradient Accumulation:** Позволяет эмулировать большой размер батча при ограниченной видеопамяти.
    * **Multiprocessing Tokenization:** Параллельная обработка данных при подготовке датасета.

### Результаты обучения:
Модель обучена на сабсете датасета `amazon_polarity` (25% данных для ускорения итераций).


### Запуск обучения:
```
# Запуск скрипта (автоматически определит GPU)
python train.py
```
### Логи обучения 

```
# trainer_state.json
{
  "best_global_step": 337500,
  "best_metric": 0.9555999715839818,
  "best_model_checkpoint": "./models/sentiment-distilbert_all/checkpoint-337500",
  "epoch": 3.0,
  "eval_steps": 500,
  "global_step": 337500,
  "is_hyper_param_search": false,
  "is_local_process_zero": true,
  "is_world_process_zero": true,
  "log_history": [
    {
      "epoch": 0.0017777777777777779,
      "grad_norm": 10.3468656539917,
      "learning_rate": 1.9900000000000003e-05,
      "loss": 0.5415,
      "step": 200
    },
    {...

    {
      "epoch": 3.0,
      "eval_accuracy": 0.9556,
      "eval_f1_macro": 0.9555999715839818,
      "eval_loss": 0.23346561193466187,
      "eval_runtime": 121.5512,
      "eval_samples_per_second": 822.698,
      "eval_steps_per_second": 51.419,
      "step": 337500
    }
  ],
  "logging_steps": 200,
  "max_steps": 337500,
  "num_input_tokens_seen": 0,
  "num_train_epochs": 3,
  "save_steps": 500}


```

## **Возможные улучшения**
Проект — пет-проект, так что он уже функционален. Но если развивать:

- Добавить аутентификацию (API keys для эндпоинтов и пользователей).
- Расширить RAG: Добавить больше коллекций (отели/продукты), улучшить промпты, реализовать полноценного агента.
- Оптимизация: Кэширование запросов, асинхронный батч в NLP.
- CI/CD: GitHub Actions для тестов/деплоя.
- Документация: Добавить примеры датасетов для Qdrant.

## Лицензия
MIT License. Свободно используйте и модифицируйте!