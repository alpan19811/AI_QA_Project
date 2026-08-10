"""Проверка подключения к PostgreSQL + схемы (pgvector)."""
import psycopg2

conn = psycopg2.connect(
    dbname="ai_qa_db",
    user="ai_qa_user",
    password="ai_qa_password",
    host="localhost",
    port=5432,
)
cur = conn.cursor()

cur.execute("SELECT extname, extversion FROM pg_extension;")
print("Extensions:", cur.fetchall())

cur.execute("SELECT to_regclass('public.documents');")
print("Table documents:", cur.fetchone()[0])

cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'documents';")
print("Indexes:", [r[0] for r in cur.fetchall()])

cur.close()
conn.close()
print("DB OK")