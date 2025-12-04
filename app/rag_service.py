# app/rag_service.py
import os
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from langsmith import Client

logger = logging.getLogger("smartreview.rag")

# Глобальная переменная для хранения агента
rag_agent = None
rag_enabled = False

# Конфигурация
from dotenv import load_dotenv
load_dotenv()

# КОНФИГУРАЦИЯ БЕРЕТСЯ ТОЛЬКО ИЗ ENV
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "amazon_complaints") # или hotel_reviews

# LangSmith конфиг (опционально)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "SmartReview_Project")
# Остальные ключи LangChain подтянутся автоматически из env, если они там есть

# описываем структуру ответа для классификатора 
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    destination: Literal["complaints", "root_cause", "comparison", "general"] = Field(
        ...,
        description="Given a user question choose to route it to one of: 'complaints', 'root_cause', 'comparison' or 'general'."
    )

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# --- Chains Builders ---

# поиск жалоб и их технический проблемы 
def build_complaint_chain(vector_store):
    # print(f" --> Story 1: поиск жалоб.")
    filter_condition = models.Filter(
        must=[models.FieldCondition(key="metadata.sentiment", match=models.MatchValue(value="negative"))]
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": filter_condition})
    
    template = """Найди конкретные детали жалоб пользователей по теме вопроса.
    Контекст (только негативные отзывы): {context}
    Вопрос: {question}
    Ответ (кратко перечисли технические проблемы):"""
    
    return (
        {"context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | get_llm()
        | StrOutputParser()
    ).with_config({"run_name": "ComplaintsChain"})

# глубинный анализ причины (только негативных отзывов) 
def build_root_cause_chain(vector_store):
    # print(f" --> Story 2: Глубинный анализ причин.")
    filter_condition = models.Filter(
        must=[models.FieldCondition(key="metadata.sentiment", match=models.MatchValue(value="negative"))]
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 8, "filter": filter_condition})
    
    template = """Ты — опытный Product Manager. Проанализируй отзывы и объясни, ПОЧЕМУ пользователи недовольны.
    Выяви паттерны (контроль качества, описание, доставка).

    Отзывы: {context}
    Тема: {question}
    
    Аналитический отчет:"""
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | get_llm()
        | StrOutputParser()
    ).with_config({"run_name": "RootChain"})

# сравнение положительных и отрицательных отзывов чтобы получить обобщенный вывод  
def build_comparative_chain(vector_store):
    # print(f" --> Story 3: Сравнение.")
    neg_filter = models.Filter(must=[models.FieldCondition(key="metadata.sentiment", match=models.MatchValue(value="negative"))])
    neg_retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": neg_filter})
    
    pos_filter = models.Filter(must=[models.FieldCondition(key="metadata.sentiment", match=models.MatchValue(value="positive"))])
    pos_retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": pos_filter})
    
    template = """Сравни опыт пользователей по теме: {topic}.


    НЕГАТИВ: {neg_context}
    ПОЗИТИВ: {pos_context}

    Задача: Таблица "Плюсы vs Минусы" и вывод по ним."""
    
    def format_list(docs):
        return "\n".join([f"- {d.page_content}" for d in docs])

    return (
        {
            "neg_context": neg_retriever | format_list,
            "pos_context": pos_retriever | format_list,
            "topic": RunnablePassthrough()
        }
        | ChatPromptTemplate.from_template(template)
        | get_llm()
        | StrOutputParser()
    ).with_config({"run_name": "ComparativeChain"})

# вызов обычной ллмки
def build_general_chain():
    # print(f" --> Story 4: обычная ЛЛМка.")
    prompt = ChatPromptTemplate.from_template(
        "Ты полезный ассистент SmartReview. Ответь пользователю вежливо, используя общие знания, но в контексте отелей.\nВопрос: {question}"
    )
    return prompt | get_llm() | StrOutputParser()

# --- Agent Initialization ---
def init_rag_agent():
    """Инициализация RAG при старте приложения"""
    global rag_agent, rag_enabled

    # --- ДОБАВИЛ ОТЛАДКУ ---
    missing_vars = []
    if not QDRANT_URL: missing_vars.append("QDRANT_URL")
    if not OPENAI_API_KEY: missing_vars.append("OPENAI_API_KEY")

    if missing_vars:
        logger.warning(f"RAG Agent SKIPPED. Missing env vars: {', '.join(missing_vars)}")
        print(f"RAG Agent SKIPPED. Missing env vars: {', '.join(missing_vars)}")
        rag_enabled = False
        return
    
    # ---
    
    if not OPENAI_API_KEY or not QDRANT_URL:
        logger.warning("RAG Agent skipped: Missing OPENAI_API_KEY or QDRANT_URL")
        rag_enabled = False
        return

    try:
        logger.info("Initializing Qdrant client...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        
        # подключаюсь к уже созданой базе 
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=COLLECTION_NAME, 
            embedding=embeddings
        )

        llm = get_llm()
        
        # Router setup
        structured_llm_router = llm.with_structured_output(RouteQuery)
        system_prompt = """You are an expert router for a hotel review analysis tool.
        Route the user's query to one of destinations:
        
        1. 'complaints': Use this when the user asks "what is wrong", "find bugs", "list issues", "complaints about X".
        2. 'root_cause': Use this when the user asks "WHY is it bad", "explain the reason", "analyze the failure".
        3. 'comparison': Use this when the user asks "pros and cons", "should I buy", "compare good and bad", "is it worth it".
        4. 'general': simple greeting or out of scope questions.
        
        Return only the destination."""
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        # цепочка классификации 
        router_chain = (router_prompt | structured_llm_router).with_config({"run_name": "RouterClassifier"}) # <-- В LangSmith будет видно это имя

        # словарь для инструментов
        chain_tools = {
            "complaints": build_complaint_chain(vector_store),
            "root_cause": build_root_cause_chain(vector_store),
            "comparison": build_comparative_chain(vector_store),
            "general": build_general_chain(),
        }

        # Функция маршрутизации (запускает нужную цепочку полученную от ЛЛМ)
        def route_logic(info):
            destination = info.destination
            logger.info(f"RAG Routing to: {destination}")
            print(f" #RAG Router decided: {destination.upper()}")
            return chain_tools[destination]

        # 4. Финальная "Магическая" цепочка
        # Сначала классифицируем -> передаем результат в route_logic -> запускаем выбранную цепочку
        rag_agent = (
            {
                "destination": router_chain,
                "question": RunnablePassthrough()
            } 
            | RunnableLambda(lambda x: route_logic(x["destination"]).invoke(x["question"]))
        )
        
        rag_enabled = True
        logger.info("RAG Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG agent: {e}")
        rag_enabled = False

# вызов RАГ агента
def query_rag(question: str) -> str:
    """Синхронная функция вызова агента"""
    if not rag_enabled or not rag_agent:
        raise RuntimeError("RAG agent is not initialized or disabled")
    return rag_agent.invoke(question)