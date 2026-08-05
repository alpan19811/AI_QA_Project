import requests
import time

# Адрес локального сервера Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# Наша модель и вопрос из банковской предметной области
MODEL = "qwen2.5"
PROMPT = "Что такое ковенантный пакет в кредитной сделке? Ответь в 3 предложениях на русском."


def ask_model(prompt: str) -> dict:
    """Отправляет запрос к Ollama и возвращает ответ + метрики скорости."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,          # ждём полный ответ, а не поток по кускам
    }

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    elapsed = time.time() - start

    response.raise_for_status()    # упадём с понятной ошибкой, если сервер не отвечает
    data = response.json()

    # Ollama возвращает служебные метрики: сколько токенов и за сколько наносекунд
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0

    return {
        "answer": data.get("response", "").strip(),
        "elapsed_sec": round(elapsed, 2),
        "tokens": eval_count,
        "tokens_per_sec": tokens_per_sec,
    }


if __name__ == "__main__":
    result = ask_model(PROMPT)

    print("=" * 60)
    print("ОТВЕТ МОДЕЛИ:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print(f"Время ответа:        {result['elapsed_sec']} сек")
    print(f"Сгенерировано токенов: {result['tokens']}")
    print(f"Скорость генерации:  {result['tokens_per_sec']} токенов/сек")