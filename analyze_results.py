"""Pandas-анализ истории eval: графики эволюции pass_rate + анализ FAIL.

Генерирует 3 PNG-графика в папке analysis/:
- 01_evolution_pass_rate.png — эволюция pass_rate по 6 итерациям
- 02_pass_rate_by_type.png — распределение PASS/FAIL по типам вопросов
- 03_top_missing_facts.png — самые частые факты, которые не находит система
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use('Agg')   # НЕ-интерактивный backend: только сохранение в файл, без окон
import matplotlib.pyplot as plt

# Настройка matplotlib для кириллицы
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPORTS_DIR = Path("results")
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

# Эталонная история — порядок важен для графика эволюции
REPORTS = [
    ("01_baseline_40pct.json",          "01 baseline\n(synth, no RAG)"),
    ("02_rag_first_87pct.json",         "02 RAG first\n(synth)"),
    ("03_final_100pct.json",            "03 eval-driven\nfix (synth)"),
    ("04_590p_dense_only_44pct.json",   "04 dense-only\n(590-P, real doc)"),
    ("05_590p_hybrid_v1_60pct.json",    "05 hybrid v1\n(590-P, AND-semantics)"),
    ("06_590p_hybrid_v2_96pct.json",    "06 hybrid v2\n(590-P, OR-semantics)"),
]


def load_reports():
    """Загружает все 6 отчётов."""
    data = []
    for filename, label in REPORTS:
        path = REPORTS_DIR / filename
        if not path.exists():
            print(f"⚠ {filename} не найден, пропускаем")
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        data.append({
            "file": filename,
            "label": label,
            "pass_rate": content.get("pass_rate_pct") or content.get("pass_rate_pct "),
            "results": content.get("results") or content.get("results "),
        })
    return data


def classify_question(case):
    """Классифицирует вопрос: answer / refuse / multi-hop."""
    q = case["question"]
    expected = case.get("expected") or case.get("expected ")
    if expected == "refuse":
        return "refuse"
    # Multi-hop: вопросы про резерв при комбинации «финансовое положение + обслуживание долга»
    if "финансовым положением" in q and "обслуживанием долга" in q:
        return "multi-hop"
    return "answer"


def plot_evolution(data):
    """График 1: эволюция pass_rate по 6 итерациям."""
    df = pd.DataFrame(data)
    # Домен: первые 3 = synthetic, последние 3 = 590-P
    df["domain"] = ["synthetic"] * 3 + ["590-P"] * 3
    colors = ['#7FB3D5' if d == "synthetic" else '#E74C3C' for d in df["domain"]]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(df)), df["pass_rate"], color=colors)

    for bar, label in zip(bars, df["label"]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.text(bar.get_x() + bar.get_width()/2, -8, label,
                 ha='center', va='top', fontsize=9)

    plt.axhline(y=80, color='green', linestyle='--', alpha=0.3, label='Target (80%)')
    plt.title("Eval-driven evolution: pass rate across 6 iterations", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Pass rate (%)", fontsize=12)
    plt.ylim(0, 115)
    plt.xticks([])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#7FB3D5', label='Synthetic credit policy'),
        Patch(facecolor='#E74C3C', label='Real regulator (CBR 590-P, 150+ pages)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    plt.tight_layout()

    path = ANALYSIS_DIR / "01_evolution_pass_rate.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def plot_by_type(data):
    """График 2: распределение PASS/FAIL по типам вопросов (итерации 04, 05, 06)."""
    rows = []
    for report in data:
        tag = report["file"][:2]
        if tag not in ["04", "05", "06"]:
            continue
        for case in report["results"]:
            qtype = classify_question(case)
            passed = case.get("deterministic_pass") or case.get("deterministic_pass ", False)
            rows.append({"iteration": tag, "question_type": qtype, "pass": bool(passed)})

    df = pd.DataFrame(rows)
    agg = df.groupby(["iteration", "question_type"])["pass"].agg(["sum", "count"]).reset_index()
    agg.columns = ["iteration", "question_type", "passed", "total"]
    agg["pass_rate"] = (agg["passed"] / agg["total"] * 100).round(1)

    pivot = agg.pivot(index="iteration", columns="question_type", values="pass_rate").fillna(0)
    for col in ["answer", "refuse", "multi-hop"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["answer", "refuse", "multi-hop"]]

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", figsize=(10, 6),
               color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.85)
    plt.title("Pass rate by question type across iterations (590-P)", fontsize=13, fontweight='bold')
    plt.ylabel("Pass rate (%)", fontsize=11)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(title="Question type", fontsize=10)
    plt.tight_layout()

    path = ANALYSIS_DIR / "02_pass_rate_by_type.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def plot_missing_facts(data):
    """График 3: топ-10 самых частых фактов, которые не находит система."""
    fact_counter = Counter()
    for report in data:
        for case in report["results"]:
            passed = case.get("deterministic_pass") or case.get("deterministic_pass ", False)
            missing = case.get("missing_facts") or case.get("missing_facts ", [])
            if not passed and missing:
                for fact_group in missing:
                    options = fact_group if isinstance(fact_group, list) else [fact_group]
                    fact_counter[options[0].strip()] += 1

    top = fact_counter.most_common(10)
    if not top:
        print("⚠ Нет missing facts для визуализации")
        return

    facts = [f[0] for f in top][::-1]
    counts = [f[1] for f in top][::-1]
    colors = ['#E74C3C' if c >= 2 else '#F39C12' for c in counts]

    plt.figure(figsize=(11, 6))
    bars = plt.barh(facts, counts, color=colors, alpha=0.85)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{count}×", ha='left', va='center', fontsize=11, fontweight='bold')

    plt.title("Most common missing facts across all FAIL cases", fontsize=13, fontweight='bold')
    plt.xlabel("Number of FAIL cases", fontsize=11)
    plt.xlim(0, max(counts) * 1.3)
    plt.tight_layout()

    path = ANALYSIS_DIR / "03_top_missing_facts.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def main():
    data = load_reports()
    print(f"Загружено отчётов: {len(data)}\n")

    plot_evolution(data)
    plot_by_type(data)
    plot_missing_facts(data)

    print("\n📈 ИТОГ:")
    for r in data:
        print(f"  {r['file']}: {r['pass_rate']}%")


if __name__ == "__main__":
    main()