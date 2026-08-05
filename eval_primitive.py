import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5"
PROMPT = "Что такое ковенантный пакет в кредитной сделке? Ответь в 3 предложениях на русском."

# Эталон: основы ключевых терминов, которые обязаны быть в корректном ответе.
# Используем основы ("заемщ", а не "заемщик"), чтобы ловить любые падежи.
EXPECTED_TERMS = ["заемщ", "кредитор", "обязательств"]


def ask_model(prompt: str) -> str:
    """Отправляет вопрос в Ollama и возвращает текст ответа."""
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def normalize(text: str) -> str:
    """Нормализация: нижний регистр + ё -> е.
    Без этого 'Заёмщиком' и 'заемщиком' были бы разными словами."""
    return text.lower().replace("ё", "е")


def check_completeness(answer: str) -> list:
    """Возвращает список смысловых блоков, которых не хватает в ответе."""
    norm = normalize(answer)
    return [term for term in EXPECTED_TERMS if term not in norm]


def find_mixed_words(answer: str) -> list:
    """Ищет слова со смешанными алфавитами — как 'kovenанты'."""
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", answer)
    mixed = []
    for w in words:
        has_latin = any(c.isascii() and c.isalpha() for c in w)
        has_cyrillic = any(not c.isascii() and c.isalpha() for c in w)
        if has_latin and has_cyrillic:
            mixed.append(w)
    return mixed


def find_foreign_phrases(answer: str) -> list:
    """Ищет последовательности из 2+ латинских слов — явные иностранные фразы.
    В ответе, который должен быть на русском, это нарушение языкового режима."""
    # [a-zA-Z]+            — первое латинское слово
    # (?:\s+[a-zA-Z]+)+    — хотя бы одно следующее латинское слово через пробел
    return re.findall(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)+", answer)


if __name__ == "__main__":
    answer = ask_model(PROMPT)
    print("ОТВЕТ МОДЕЛИ:\n", answer)
    print("=" * 60)

    missing = check_completeness(answer)
    mixed = find_mixed_words(answer)
    foreign = find_foreign_phrases(answer)

    print("ПРОВЕРКА ПОЛНОСТИ:")
    print(f"  FAIL — не хватает: {missing}" if missing else "  PASS — все ключевые термины найдены")

    print("ПРОВЕРКА ЧИСТОТЫ ПИСЬМА (смешение алфавитов в слове):")
    print(f"  FAIL — смешанные слова: {mixed}" if mixed else "  PASS — смешанных слов нет")

    print("ПРОВЕРКА ЯЗЫКОВОЙ ЧИСТОТЫ (иностранные фразы):")
    print(f"  FAIL — иностранные фразы: {foreign}" if foreign else "  PASS — иностранных фраз нет")

    verdict = "PASS" if not missing and not mixed and not foreign else "FAIL"
    print("=" * 60)
    print(f"ИТОГОВЫЙ ВЕРДИКТ: {verdict}")