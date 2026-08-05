import requests
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"   # модель для эмбеддингов (векторов)
CHAT_MODEL = "qwen2.5"             # модель для генерации ответа

# ── 1. ИСТОЧНИК: синтетический фрагмент кредитной политики (ваш домен) ──
DOCUMENT = """
Кредитная политика банка в части ковенантного пакета (фрагмент).

1. Ковенантный пакет — набор финансовых и нефинансовых обязательств заёмщика, контролируемых кредитором в течение срока действия кредита.

2. Финансовые ковенанты: коэффициент Debt/EBITDA не должен превышать 3.0x; коэффициент DSCR (покрытие долга) должен быть не ниже 1.2.

3. Ограничение дивидендов: заёмщик вправе направлять на выплату дивидендов не более 50% чистой прибыли за отчётный год.

4. Отчётность: заёмщик предоставляет финансовую отчётность ежеквартально, не позднее 45 календарных дней после окончания квартала.

5. Последствия нарушения: при нарушении любого ковенанта кредитор вправе потребовать досрочного погашения кредита.
"""

# ── 2. ЭМБЕДДИНГИ: просим Ollama превратить текст в вектор ──
def get_embedding(text: str) -> list:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


# ── 3. ГЕНЕРАЦИЯ: просим qwen2.5 ответить ТОЛЬКО по контексту ──
def generate(context: str, question: str) -> str:
    prompt = (
        "Ответь на вопрос, используя ТОЛЬКО контекст ниже. "
        "Если в контексте нет ответа — напиши 'В документе этого нет'.\n"
        f"Контекст:\n{context}\nВопрос: {question}"
    )
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def build_index(document: str):
    """Чанкинг → эмбеддинги → векторная БД."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(document)

    client = chromadb.Client()  # in-memory БД (для демо)
    collection = client.get_or_create_collection("credit_policy")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=[get_embedding(c) for c in chunks],
    )
    return collection, len(chunks)


def retrieve(collection, question: str, k: int = 2) -> str:
    """Поиск k самых похожих чанков по смыслу вопроса."""
    q_emb = get_embedding(question)
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return "\n".join(res["documents"][0])


if __name__ == "__main__":
    print("Строим индекс (чанкинг + эмбеддинги)...")
    collection, n = build_index(DOCUMENT)
    print(f"Готово. Чанков в базе: {n}\n")

    questions = [
        "Какой максимальный коэффициент Debt/EBITDA допустим по кредитной политике?",  # ответ ЕСТЬ в документе
        "Какая процентная ставка по кредиту установлена политикой?",                    # ответа НЕТ в документе
    ]

    for q in questions:
        print("=" * 60)
        print("ВОПРОС:", q)
        context = retrieve(collection, q)
        print("ИЗВЛЕЧЁННЫЙ КОНТЕКСТ:\n", context, "\n")
        print("ОТВЕТ МОДЕЛИ:\n", generate(context, q), "\n")