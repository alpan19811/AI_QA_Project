import json
import os
import re
from datetime import datetime

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5"
PROMPT = "Что такое ковенантный пакет в кредитной сделке? Ответь в 3 предложениях на русском."

EXPECTED_TERMS = ["заемщ", "кредитор", "обязательств"]


def ask_model(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def check_completeness(answer: str) -> list:
    norm = normalize(answer)
    return [t for t in EXPECTED_TERMS if t not in norm]


def find_mixed_words(answer: str) -> list:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", answer)
    return [
        w for w in words
        if any(c.isascii() and c.isalpha() for c in w)
        and any(not c.isascii() and c.isalpha() for c in w)
    ]


def find_foreign_phrases(answer: str) -> list:
    return re.findall(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)+", answer)


def run_single_eval(prompt: str) -> dict:
    """ОДИН прогон: спросить модель + прогнать все проверки."""
    answer = ask_model(prompt)
    return {
        "answer": answer,
        "checks": {
            "completeness": check_completeness(answer),
            "script_purity": find_mixed_words(answer),
            "language_purity": find_foreign_phrases(answer),
        },
    }


def run_batch(n_runs: int) -> dict:
    """Цикл из n_runs прогонов + агрегация статистики."""
    results = []
    for i in range(1, n_runs + 1):
        single = run_single_eval(PROMPT)
        # каждая проверка вернула СПИСОК проблем; пустой список = PASS
        verdicts = {name: not issues for name, issues in single["checks"].items()}
        overall = all(verdicts.values())
        results.append({
            "run": i,
            "answer": single["answer"],
            "checks": single["checks"],
            "verdicts": verdicts,
            "overall": overall,
        })
        print(f"Прогон {i:>2}: {'PASS' if overall else 'FAIL'}")

    passed = sum(1 for r in results if r["overall"])
    check_failures = {
        name: sum(1 for r in results if not r["verdicts"][name])
        for name in ["completeness", "script_purity", "language_purity"]
    }

    return {
        "model": MODEL,
        "prompt": PROMPT,
        "n_runs": n_runs,
        "passed": passed,
        "pass_rate_pct": round(passed / n_runs * 100, 1),
        "check_failures": check_failures,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }


def save_report(report: dict) -> str:
    """Сохранить отчёт в results/eval_report_<время>.json."""
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"eval_report_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    report = run_batch(n_runs=5)

    print("=" * 60)
    print(f"Модель: {report['model']}")
    print(f"Прогонов: {report['n_runs']}, прошло: {report['passed']}")
    print(f"PASS_RATE: {report['pass_rate_pct']}%")
    print("-" * 60)
    print("Падения по каждой проверке:")
    for name, failed in report["check_failures"].items():
        print(f"  {name}: {failed} из {report['n_runs']}")
    print("=" * 60)
    print(f"Отчёт сохранён: {save_report(report)}")