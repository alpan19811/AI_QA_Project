"""Батч-оценка RAG-ассистента по Golden Dataset (Положение 590-П).

Два слоя проверок:
  1. Детерминированные (быстро): факты по основам/any-of + детекция отказа;
  2. LLM-as-a-Judge (медленно, DeepEval): Faithfulness + Answer Relevancy.
     Отключается флагом USE_JUDGE = False для быстрых итераций.
"""

import json
import os
from datetime import datetime

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from eval_batch import normalize
from golden_dataset_590p import GOLDEN_DATASET
from rag_assistant import DOCUMENT, build_index, generate, get_connection, retrieve
from rag_eval import OllamaJudge, safe_measure

USE_JUDGE = False  # True — включить DeepEval-судью (медленно, ~10 мин)

REFUSAL_MARKERS = ["в документе этого нет", "нет в документе", "не содержится"]


def check_facts(answer: str, facts: list) -> list:
    """Возвращает список отсутствующих смысловых блоков. list внутри = any-of (синонимы)."""
    norm = normalize(answer)
    missing = []
    for f in facts:
        options = f if isinstance(f, list) else [f]
        if not any(opt in norm for opt in options):
            missing.append(options)
    return missing


def is_refusal(answer: str) -> bool:
    norm = normalize(answer)
    return any(m in norm for m in REFUSAL_MARKERS)


def eval_case(case: dict, conn, judge) -> dict:
    question = case["question"]
    context = retrieve(conn, question, k=5)
    answer = generate(context, question)

    refused = is_refusal(answer)
    missing = check_facts(answer, case["facts"])

    if case["expected"] == "answer":
        det_ok = (not missing) and (not refused)   # ответил И привёл факты
    else:
        det_ok = refused                            # честно отказался

    result = {
        "question": question,
        "expected": case["expected"],
        "context": context,
        "answer": answer,
        "refused": refused,
        "missing_facts": missing,
        "deterministic_pass": det_ok,
    }

    if USE_JUDGE and case["expected"] == "answer":
        tc = LLMTestCase(input=question, actual_output=answer, retrieval_context=[context])
        f_score, f_ok, _ = safe_measure(FaithfulnessMetric(threshold=0.7, model=judge), tc, "Faithfulness")
        r_score, r_ok, _ = safe_measure(AnswerRelevancyMetric(threshold=0.7, model=judge), tc, "Relevancy")
        result["faithfulness"] = f_score
        result["relevancy"] = r_score
        result["judge_pass"] = bool(f_ok and r_ok)
    return result


def main():
    # Персистентный индекс: пересобираем только если таблица пуста
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents;")
    n = cur.fetchone()[0]
    cur.close()

    if n == 0:
        print("Индекс пуст — строим...")
        conn, n = build_index(DOCUMENT)
    print(f"Чанков в индексе: {n}\n")

    judge = OllamaJudge() if USE_JUDGE else None

    results = []
    for i, case in enumerate(GOLDEN_DATASET, 1):
        print(f"[{i}/{len(GOLDEN_DATASET)}] {case['question']}")
        res = eval_case(case, conn, judge)
        results.append(res)
        status = "PASS" if res["deterministic_pass"] else "FAIL"
        extra = ""
        if "judge_pass" in res:
            extra = f" | judge: F={res['faithfulness']} R={res['relevancy']} -> {'PASS' if res['judge_pass'] else 'FAIL'}"
        print(f"   -> {status}{extra}\n")

    passed = sum(1 for r in results if r["deterministic_pass"])
    pass_rate = round(passed / len(results) * 100, 1)

    print("=" * 70)
    print(f"Кейсов: {len(results)}, прошло: {passed}")
    print(f"PASS_RATE (детерминированный): {pass_rate}%")

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"rag_batch_590p_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pass_rate_pct": pass_rate, "n_cases": len(results),
                   "use_judge": USE_JUDGE, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {path}")


if __name__ == "__main__":
    main()