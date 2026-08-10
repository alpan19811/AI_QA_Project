"""Первый векторный round-trip: Ollama embeddings -> pgvector -> cosine search."""
import requests
import psycopg2

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

DB_CONFIG = dict(
    dbname="ai_qa_db",
    user="ai_qa_user",
    password="ai_qa_password",
    host="localhost",
    port=5432,
)


def get_embedding(text: str) -> list:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def main():
    texts = [
        "Ковенантный пакет — набор обязательств заёмщика перед кредитором.",
        "Debt/EBITDA не должен превышать 3.0x.",
        "Заёмщик предоставляет отчётность ежеквартально, не позднее 45 дней.",
    ]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("DELETE FROM documents;")  # чистим от прошлых прогонов

    for t in texts:
        emb = get_embedding(t)
        cur.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s::vector);",
            (t, "roundtrip_test", emb),
        )
    conn.commit()
    print(f"Inserted {len(texts)} chunks")

    # Вопрос сформулирован ИНАЧЕ, чем текст — проверяем семантику, не ключевые слова
    question = "Какой максимум по коэффициенту долг к EBITDA разрешён?"
    q_emb = get_embedding(question)

    cur.execute(
        """
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 2;
        """,
        (q_emb, q_emb),
    )
    print("\nQuestion:", question)
    for content, sim in cur.fetchall():
        print(f"  sim={sim:.3f} | {content}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()