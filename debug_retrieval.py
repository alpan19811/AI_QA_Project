"""Диагностика retrieval: что реально находит поиск по вопросу."""
import psycopg2
from rag_assistant import get_embedding, DB_CONFIG

questions = [
    "Какой размер резерва установлен для стандартных ссуд (I категория качества)?",
    "Как классифицируется ссуда, если финансовое положение заемщика хорошее, а обслуживание долга среднее?",
]

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

for q in questions:
    q_emb = get_embedding(q)
    cur.execute(
        """
        SELECT content, 1 - (embedding <=> %s::vector) AS sim
        FROM documents ORDER BY embedding <=> %s::vector LIMIT 3;
        """,
        (q_emb, q_emb),
    )
    print("=" * 70)
    print("ВОПРОС:", q)
    for content, sim in cur.fetchall():
        print(f"\n--- sim={sim:.3f} ---")
        print(content[:600])

conn.close()