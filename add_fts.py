"""Миграция: колонка полнотекстового поиска (русский стеммер) для гибридного поиска.

tsvector — GENERATED-колонка: считается автоматически при каждом INSERT,
поддерживать вручную не нужно.
"""
from rag_assistant import get_connection

conn = get_connection()
cur = conn.cursor()
cur.execute("""
    ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED;
""")
cur.execute("CREATE INDEX IF NOT EXISTS documents_tsv_idx ON documents USING gin(tsv);")
conn.commit()
cur.close()
conn.close()
print("FTS-колонка + GIN-индекс готовы")