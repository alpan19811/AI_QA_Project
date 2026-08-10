-- Включаем векторное расширение
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица документов с эмбеддингами
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT,
    embedding vector(768)   -- 768 = размерность nomic-embed-text
);

-- HNSW-индекс для быстрого поиска ближайших векторов
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents USING hnsw (embedding vector_cosine_ops);