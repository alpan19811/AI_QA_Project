"""Загрузка Положения ЦБ РФ 590-П: скачивание PDF + парсинг в текст."""

import requests
import pdfplumber
from pathlib import Path

# Основной документ Положения 590-П (полный текст)
PDF_URL = "https://www.cbr.ru/Crosscut/LawActs/File/12209"
PDF_PATH = Path("documents/590-P.pdf")
TXT_PATH = Path("documents/590-P.txt")


def download_pdf():
    """Скачиваем PDF с сайта ЦБ РФ."""
    PDF_PATH.parent.mkdir(exist_ok=True)
    print(f"Скачиваем {PDF_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    r = requests.get(PDF_URL, timeout=60, headers=headers)
    r.raise_for_status()
    PDF_PATH.write_bytes(r.content)
    print(f"Сохранено: {PDF_PATH} ({len(r.content)} bytes)")


def parse_pdf_to_text():
    """Извлекаем текст из PDF, страница за страницей."""
    print(f"Парсим {PDF_PATH}...")
    with pdfplumber.open(PDF_PATH) as pdf:
        pages_text = []
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                pages_text.append(f"\n--- Страница {i} ---\n{text}")
        full_text = "\n".join(pages_text)

    TXT_PATH.write_text(full_text, encoding="utf-8")
    print(f"Извлечено {len(pages_text)} страниц, {len(full_text)} символов")
    print(f"Сохранено: {TXT_PATH}")
    return full_text


if __name__ == "__main__":
    download_pdf()
    text = parse_pdf_to_text()
    print("\nПервые 500 символов:")
    print(text[:500])