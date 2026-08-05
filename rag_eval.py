"""Оценка RAG через DeepEval с локальным судьёй (LLM-as-a-Judge).

Недели 3–5 трека AI Evaluation Engineer.
Оценивает RAG-ассистента (rag_assistant.py) по метрикам Faithfulness и
Answer Relevancy, используя локальную qwen2.5 в роли судьи — офлайн,
без API-ключей (важно для банков: данные не уходят наружу).
"""

import requests
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from rag_assistant import DOCUMENT, build_index, retrieve, generate

OLLAMA_URL = "http://localhost:11434"


class OllamaJudge(DeepEvalBaseLLM):
    """LLM-as-a-Judge: локальная qwen2.5 в роли судьи с контролем формата."""

    def load_model(self):
        return self

    def generate(self, prompt: str, **kwargs) -> str:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "format": "json",          # 1. принудительный валидный JSON
                "system": (                # 2. явная инструкция следовать схеме
                    "You are a strict evaluation judge. "
                    "Respond ONLY with valid JSON that exactly matches the requested schema. "
                    "No markdown, no comments, no extra text."
                ),
                "options": {"temperature": 0},   # 3. детерминированный судья
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["response"]

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt)   # локально проще синхронно

    def get_model_name(self) -> str:
        return "qwen2.5 (local Ollama judge, JSON-mode)"


def safe_measure(metric, test_case, metric_name: str, attempts: int = 2):
    """Защита от падения судьи: повтор + graceful fallback вместо краха."""
    last_err = ""
    for _ in range(attempts):
        try:
            metric.measure(test_case)
            return metric.score, metric.success, metric.reason
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    print(f"  [WARN] {metric_name}: судья не смог оценить ({last_err})")
    return None, False, f"judge error: {last_err}"


def main():
    print("Строим индекс...")
    collection, n = build_index(DOCUMENT)
    print(f"Чанков: {n}\n")

    questions = [
        "Какой максимальный коэффициент Debt/EBITDA допустим по кредитной политике?",
        "Какая процентная ставка по кредиту установлена политикой?",
    ]

    judge = OllamaJudge()

    for q in questions:
        print("=" * 70)
        print("ВОПРОС:", q)
        context = retrieve(collection, q)
        answer = generate(context, q)
        print("ОТВЕТ:", answer, "\n")

        test_case = LLMTestCase(
            input=q,
            actual_output=answer,
            retrieval_context=[context],   # то, на что модель должна опираться
        )

        f_score, f_ok, f_reason = safe_measure(
            FaithfulnessMetric(threshold=0.7, model=judge), test_case, "Faithfulness"
        )
        r_score, r_ok, r_reason = safe_measure(
            AnswerRelevancyMetric(threshold=0.7, model=judge), test_case, "Answer Relevancy"
        )

        for name, score, ok, reason in (
            ("Faithfulness", f_score, f_ok, f_reason),
            ("Answer Relevancy", r_score, r_ok, r_reason),
        ):
            if score is None:
                print(f"{name}: N/A — судья не смог оценить (edge case отказа)")
            else:
                verdict = "PASS" if ok else "FAIL"
                print(f"{name}: {score:.2f} (порог 0.7) -> {verdict} | {reason}")
        print()


if __name__ == "__main__":
    main()