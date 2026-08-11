"""Unit-тесты RAG-ассистента: НЕ требуют PostgreSQL/Ollama (всё в моках).

Запуск: pytest -v -m unit
"""

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_check_facts_finds_present_facts():
    from rag_eval_batch import check_facts
    answer = "Для стандартных ссуд резерв 0 процентов"
    assert check_facts(answer, ["0", "стандартн"]) == []


@pytest.mark.unit
def test_check_facts_detects_missing_facts():
    """check_facts находит отсутствующие факты (подстроки в нижнем регистре)."""
    from rag_eval_batch import check_facts
    answer = "Резерв составляет 20 процентов"
    missing = check_facts(answer, ["20", "100"])
    assert missing == [["100"]]


@pytest.mark.unit
def test_check_facts_any_of_semantics():
    """check_facts: any-of засчитывается, если есть ХОТЯ БЫ ОДИН вариант."""
    from rag_eval_batch import check_facts
    answer = "Резерв составляет пять процентов"
    assert check_facts(answer, [["пять", "5"]]) == []


@pytest.mark.unit
def test_is_refusal_detects_refusal_phrases():
    from rag_eval_batch import is_refusal
    assert is_refusal("В документе этого нет")
    assert is_refusal("Нет в документе такой информации")
    assert not is_refusal("Резерв составляет 20 процентов")


@pytest.mark.unit
def test_classify_question_answer_type():
    from analyze_results import classify_question
    assert classify_question({"question": "Сколько категорий?", "expected": "answer"}) == "answer"
    assert classify_question({"question": "Какой курс доллара?", "expected": "refuse"}) == "refuse"


@pytest.mark.unit
def test_classify_question_multi_hop():
    from analyze_results import classify_question
    case = {
        "question": "Какой размер расчетного резерва нужен для ссуды с хорошим финансовым положением и хорошим обслуживанием долга?",
        "expected": "answer",
    }
    assert classify_question(case) == "multi-hop"


@pytest.mark.unit
def test_eval_case_answer_success():
    from rag_eval_batch import eval_case
    case = {"question": "Какой резерв?", "expected": "answer", "facts": ["20"]}
    with patch("rag_eval_batch.retrieve", return_value="Резерв 20 процентов"), \
         patch("rag_eval_batch.generate", return_value="Резерв составляет 20 процентов"):
        result = eval_case(case, conn=None, judge=None)
    assert result["deterministic_pass"] is True
    assert result["refused"] is False
    assert result["missing_facts"] == []


@pytest.mark.unit
def test_eval_case_refuse_success():
    from rag_eval_batch import eval_case
    case = {"question": "Какой курс доллара?", "expected": "refuse", "facts": []}
    with patch("rag_eval_batch.retrieve", return_value="какой-то контекст"), \
         patch("rag_eval_batch.generate", return_value="В документе этого нет"):
        result = eval_case(case, conn=None, judge=None)
    assert result["deterministic_pass"] is True
    assert result["refused"] is True


@pytest.mark.unit
def test_eval_case_wrong_refuse_fails():
    from rag_eval_batch import eval_case
    case = {"question": "Какой курс доллара?", "expected": "refuse", "facts": []}
    with patch("rag_eval_batch.retrieve", return_value="контекст"), \
         patch("rag_eval_batch.generate", return_value="Курс доллара 90 рублей"):
        result = eval_case(case, conn=None, judge=None)
    assert result["deterministic_pass"] is False
    assert result["refused"] is False


@pytest.mark.unit
def test_normalize_lowercases_and_replaces_yo():
    """Контракт normalize: нижний регистр + ё→е; пунктуация и пробелы СОХРАНЯЮТСЯ."""
    from eval_batch import normalize
    assert normalize("Резерв: 20% Ёлка!") == "резерв: 20% елка!"
