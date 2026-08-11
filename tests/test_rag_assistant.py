"""Базовые pytest-тесты для RAG-ассистента по Положению 590-П.

Smoke-тесты: быстрые проверки критичных компонентов без реальных LLM-вызовов.
Запуск: pytest -v
"""

import psycopg2
from rag_assistant import (
    get_connection,
    retrieve,
    get_embedding,
)

DB_CONFIG = dict(
    dbname="ai_qa_db",
    user="ai_qa_user",
    password="ai_qa_password",
    host="localhost",
    port=5432,
)


def test_db_connection():
    """Подключение к PostgreSQL работает."""
    conn = get_connection()
    assert conn is not None
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    assert cur.fetchone()[0] == 1
    cur.close()
    conn.close()


def test_retrieve_returns_chunks():
    """Retrieval возвращает минимум 1 чанк."""
    conn = get_connection()
    context = retrieve(conn, "Какой размер резерва для стандартных ссуд?", k=3)
    conn.close()
    assert len(context) > 0, "Retrieval должен вернуть хотя бы 1 чанк"
    assert "процент" in context.lower() or "резерв" in context.lower()


def test_table_propositions_in_index():
    """Сериализованные таблицы (пропозиции) загружены в индекс."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents WHERE source = '590-P-tables';")
    n_tables = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert n_tables >= 23, f"Должно быть ≥23 табличных чанков, найдено {n_tables}"


def test_hybrid_search_finds_tables():
    """Гибридный поиск находит табличные данные (multi-hop)."""
    conn = get_connection()
    # Вопрос, требующий Таблицу 1 + Таблицу 2
    context = retrieve(
        conn,
        "Какой размер расчетного резерва нужен для ссуды с хорошим финансовым положением и хорошим обслуживанием долга?",
        k=3,
    )
    conn.close()
    assert "0 процентов" in context or "стандартн" in context.lower(), (
        "Гибридный поиск должен найти табличный ответ (0% для хорошего+хорошего)"
    )


def test_refusal_mechanism():
    """Refusal работает: retrieval возвращает нерелевантный контекст для офтопик-вопроса."""
    conn = get_connection()
    # Вопрос, на который ответа в 590-П нет
    context = retrieve(
        conn,
        "Какой курс доллара США установлен Банком России на текущую дату?",
        k=3,
    )
    conn.close()
    # Контекст не должен содержать конкретных цифр курса (т.к. в документе их нет)
    # Это проверяет, что retrieval не галлюцинирует нерелевантные данные
    assert "курс" not in context.lower() or "доллар" not in context.lower() or len(context) < 500, (
        "Retrieval не должен возвращать нерелевантный контекст для офтопик-вопроса"
    )