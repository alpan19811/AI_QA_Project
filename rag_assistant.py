"""RAG-пайплайн на PostgreSQL + pgvector по полному тексту Положения ЦБ РФ 590-П.

Цепочка: чанкинг -> эмбеддинги (nomic-embed-text) -> pgvector -> retrieval -> генерация.
Индекс персистентный: не пересобирается при каждом запуске, строится только при пустой таблице.
Наблюдаемость: opt-in трейсинг через LangSmith (включается переменными окружения).
"""

import time
from pathlib import Path

import requests
import psycopg2
from dotenv import load_dotenv
from langsmith import traceable
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tables_590p import TABLE_SENTENCES

# Секреты из .env (не в git); переменные ОС (например, секреты CI) имеют приоритет
load_dotenv(Path(__file__).parent / ".env")

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5"

DB_CONFIG = dict(
    dbname="ai_qa_db",
    user="ai_qa_user",
    password="ai_qa_password",
    host="localhost",
    port=5432,
)

DOC_PATH = Path(__file__).parent / "documents" / "doc_590-P.txt"
DOC_SOURCE = "590-P"

# Юридический документ: длинные чанки с перекрытием, чтобы смысл пункта не рвался
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_document() -> str:
    """Читает полный текст Положения 590-П из файла."""
    return DOC_PATH.read_text(encoding="utf-8")


# Совместимость с импортами rag_eval.py / rag_eval_batch.py
DOCUMENT = load_document()


@traceable(name="embedding.nomic-embed-text")
def get_embedding(text: str) -> list:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def build_index(document: str, source: str = DOC_SOURCE):
    """Чанкинг -> эмбеддинги -> INSERT в pgvector. Возвращает (conn, n_chunks)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = [c.strip() for c in splitter.split_text(document) if len(c.strip()) > 50]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents;")
    for i, c in enumerate(chunks, 1):
        cur.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s::vector);",
            (c, source, get_embedding(c)),
        )
        if i % 50 == 0:
            conn.commit()
            print(f"  ...заэмбеддено {i}/{len(chunks)}")

    # Сериализованные таблицы (пропозиции) — слой, «понимающий» вопросы
    for s in TABLE_SENTENCES:
        cur.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s::vector);",
            (s, "590-P-tables", get_embedding(s)),
        )

    conn.commit()
    cur.close()
    return conn, len(chunks) + len(TABLE_SENTENCES)


def retrieve(conn, question: str, k: int = 3, deep: int = 30) -> str:
    """Гибридный поиск: вектор (pgvector) + полный текст (tsvector, русский),
    слияние через Reciprocal Rank Fusion (RRF).

    FTS-ветка с OR-семантикой: plainto_tsquery по умолчанию требует ВСЕХ слов
    (AND), из-за чего перефразированные вопросы не находили пункты.
    OR + ts_rank ранжирует по числу совпавших основ.

    Примечание: не декорируем @traceable — psycopg2-коннектор несериализуем
    для LangSmith. Дочерние спаны get_embedding всё равно отследятся автоматически.
    """
    q_emb = get_embedding(question)
    cur = conn.cursor()

    # Ветка 1: семантика
    cur.execute(
        "SELECT id, content FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;",
        (q_emb, deep),
    )
    vec_rows = cur.fetchall()

    # Ветка 2: лексика (OR-семантика)
    cur.execute("SELECT plainto_tsquery('russian', %s)::text;", (question,))
    or_query = (cur.fetchone()[0] or "").replace(" & ", " | ")
    fts_rows = []
    if or_query.strip():
        cur.execute(
            """
            SELECT id, content FROM documents
            WHERE tsv @@ to_tsquery('russian', %s)
            ORDER BY ts_rank(tsv, to_tsquery('russian', %s), 32) DESC
            LIMIT %s;
            """,
            (or_query, or_query, deep),
        )
        fts_rows = cur.fetchall()
    cur.close()

    # RRF-фьюжн: 1/(60+rank) из обеих веток
    scores, contents = {}, {}
    for rank, (did, c) in enumerate(vec_rows, 1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (60 + rank)
        contents[did] = c
    for rank, (did, c) in enumerate(fts_rows, 1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (60 + rank)
        contents[did] = c

    top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return "\n\n".join(contents[d] for d in top_ids)


@traceable(name="generation.qwen2.5")
def generate(context: str, question: str) -> str:
    prompt = (
        "Ответь на вопрос, используя ТОЛЬКО контекст ниже (фрагмент Положения ЦБ РФ 590-П). "
        "Если в контексте нет ответа — напиши 'В документе этого нет'.\n"
        f"Контекст:\n{context}\nВопрос: {question}"
    )
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


@traceable(name="rag.pipeline")
def answer(question: str, k: int = 3) -> dict:
    """Полный RAG-цикл одним трейсируемым вызовом: retrieve -> generate.

    В LangSmith виден как корневой спан с дочерними:
      rag.pipeline
      ├── embedding.nomic-embed-text  (latency ~50-100ms)
      └── generation.qwen2.5          (latency ~2-4s)
    """
    conn = get_connection()
    try:
        context = retrieve(conn, question, k=k)
        result = generate(context, question)
    finally:
        conn.close()
    return {"context": context, "answer": result}


if __name__ == "__main__":
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents;")
    n = cur.fetchone()[0]
    cur.close()

    if n == 0:
        print("Индекс пуст — строим из Положения 590-П...")
        t0 = time.time()
        conn, n = build_index(DOCUMENT)
        print(f"Чанков: {n} | время индексации: {time.time() - t0:.0f} сек\n")
        conn.close()
    else:
        print(f"Используем существующий индекс: {n} чанков\n")

    questions = [
        "Какой размер резерва установлен для стандартных ссуд (I категория качества)?",
        "Как классифицируется ссуда, если финансовое положение заемщика хорошее, а обслуживание долга среднее?",
        "Какая максимальная процентная ставка по потребительскому кредиту установлена Положением?",
    ]
    for q in questions:
        print("=" * 70)
        print("ВОПРОС:", q)
        res = answer(q)
        print("ОТВЕТ:\n", res["answer"], "\n")