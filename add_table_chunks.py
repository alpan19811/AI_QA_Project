"""Инкрементально добавляет сериализованные таблицы в индекс.
Идемпотентно: сначала удаляет старые табличные чанки."""
from rag_assistant import get_connection, get_embedding
from tables_590p import TABLE_SENTENCES

conn = get_connection()
cur = conn.cursor()
cur.execute("DELETE FROM documents WHERE source = '590-P-tables';")
for s in TABLE_SENTENCES:
    cur.execute(
        "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s::vector);",
        (s, "590-P-tables", get_embedding(s)),
    )
conn.commit()
cur.close()
conn.close()
print(f"Добавлено {len(TABLE_SENTENCES)} табличных чанков")