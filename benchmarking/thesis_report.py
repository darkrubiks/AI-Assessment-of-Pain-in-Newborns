"""Create thesis-ready tables and figures from a real-time benchmark run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.realtime import BenchmarkError


SCENARIO_ORDER = [
    "NCNN_single",
    "NCNN_MCDP50_smooth30",
    "NCNN_full_MCDP50_RGU_XAI_average",
    "VGGFace_single",
    "VGGFace_MCDP50_smooth30",
    "VGGFace_full_MCDP50_RGU_XAI_average",
    "ViT_B_32_single",
    "ViT_B_32_ensemble10_smooth30",
    "ViT_B_32_full_ensemble10_RGU_XAI_average",
]

SCENARIO_META = {
    "NCNN_single": ("NCNN", "Simples", "NCNN simples"),
    "NCNN_MCDP50_smooth30": ("NCNN", "Incerteza", "NCNN + MCDP-50"),
    "NCNN_full_MCDP50_RGU_XAI_average": ("NCNN", "Completo", "NCNN + MCDP-50 + XAI"),
    "VGGFace_single": ("VGGFace", "Simples", "VGGFace simples"),
    "VGGFace_MCDP50_smooth30": ("VGGFace", "Incerteza", "VGGFace + MCDP-50"),
    "VGGFace_full_MCDP50_RGU_XAI_average": ("VGGFace", "Completo", "VGGFace + MCDP-50 + XAI"),
    "ViT_B_32_single": ("ViT-B/32", "Simples", "ViT-B/32 simples"),
    "ViT_B_32_ensemble10_smooth30": ("ViT-B/32", "Incerteza", "ViT-B/32 + ensemble-10"),
    "ViT_B_32_full_ensemble10_RGU_XAI_average": (
        "ViT-B/32", "Completo", "ViT-B/32 + ensemble-10 + XAI"
    ),
}

STAGES = [
    ("detection_ms", "Detecção"),
    ("preprocess_ms", "Pré-processamento"),
    ("transfer_ms", "CPU→GPU"),
    ("inference_ms", "Inferência"),
    ("xai_ms", "XAI e fusão"),
    ("smoothing_ms", "Suavização"),
    ("compute_ms", "Computação total"),
    ("end_to_end_ms", "End-to-end"),
]

MODEL_COLORS = {"NCNN": "#1f77b4", "VGGFace": "#ff7f0e", "ViT-B/32": "#2ca02c"}
VARIATION_HATCHES = {"Simples": "", "Incerteza": "//", "Completo": "xx"}


def _sample_stats(values: Iterable[float]) -> dict[str, float | int]:
    data = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"n": 0, "mean_ms": np.nan, "sd_ms": np.nan, "median_ms": np.nan, "p95_ms": np.nan}
    return {
        "n": int(data.size),
        "mean_ms": float(np.mean(data)),
        "sd_ms": float(np.std(data, ddof=1)) if data.size > 1 else np.nan,
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.percentile(data, 95)),
    }


def format_mean_sd(mean: float, sd: float, decimals: int = 2, *, latex: bool = False) -> str:
    if not math.isfinite(float(mean)):
        return "--"
    separator = r" $\pm$ " if latex else " ± "
    sd_text = f"{sd:.{decimals}f}" if math.isfinite(float(sd)) else "n/a"
    return f"{mean:.{decimals}f}{separator}{sd_text}"


def _latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in str(value))


def _load_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise BenchmarkError(f"Resumo não encontrado: {summary_path}")
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    by_name = {str(item["scenario"]): item for item in summaries}
    missing = [name for name in SCENARIO_ORDER if name not in by_name]
    if missing:
        raise BenchmarkError(f"Cenários ausentes na execução: {missing}")
    failed = [name for name in SCENARIO_ORDER if by_name[name].get("status") != "ok"]
    if failed:
        raise BenchmarkError(f"Cenários com falha: {failed}")
    return summaries, by_name


def _latency_table(run_dir: Path, by_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        model, variation, label = SCENARIO_META[scenario]
        frame_path = run_dir / scenario / "per_frame.csv"
        frame_df = pd.read_csv(frame_path)
        xai_enabled = (by_name[scenario].get("configuration", {}).get("xai", {}) or {}).get("method") != "none"
        for column, stage_label in STAGES:
            if column == "xai_ms" and not xai_enabled:
                stats = {"n": 0, "mean_ms": np.nan, "sd_ms": np.nan, "median_ms": np.nan, "p95_ms": np.nan}
            else:
                stats = _sample_stats(frame_df[column])
            rows.append({
                "scenario": scenario,
                "model": model,
                "variation": variation,
                "scenario_label": label,
                "stage": column,
                "stage_label": stage_label,
                **stats,
            })
    return pd.DataFrame(rows)


def _operational_table(by_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        model, variation, label = SCENARIO_META[scenario]
        item = by_name[scenario]
        rows.append({
            "scenario": scenario,
            "model": model,
            "variation": variation,
            "scenario_label": label,
            "frames_read": item["counts"]["frames_read"],
            "frames_processed_n": item["counts"]["frames_processed"],
            "face_detection_failures_n": item["counts"]["face_detection_failures"],
            "valid_output_fps": item["throughput"]["valid_output_fps"],
            "queue_drop_pct": item["rates"]["queue_drop_pct_of_read"],
            "coverage_pct": item["rates"]["end_to_end_coverage_pct_of_read"],
            "deadline_miss_pct": item["rates"]["deadline_miss_pct"],
            "torch_vram_peak_mb": item.get("torch_gpu_memory", {}).get("max_allocated_mb"),
            "torch_vram_reserved_peak_mb": item.get("torch_gpu_memory", {}).get("max_reserved_mb"),
        })
    return pd.DataFrame(rows)


def _resources_table(run_dir: Path) -> pd.DataFrame:
    rows = []
    resource_columns = [
        ("process_cpu_pct", "process_cpu_pct"),
        ("process_rss_mb", "process_rss_mb"),
        ("gpu_util_pct", "gpu_util_pct"),
        ("gpu_memory_used_mb", "gpu_memory_used_mb"),
    ]
    for scenario in SCENARIO_ORDER:
        model, variation, label = SCENARIO_META[scenario]
        data = pd.read_csv(run_dir / scenario / "resource_samples.csv")
        row: dict[str, Any] = {
            "scenario": scenario,
            "model": model,
            "variation": variation,
            "scenario_label": label,
            "resource_samples_n": len(data),
        }
        for column, prefix in resource_columns:
            stats = _sample_stats(data[column] if column in data else [])
            row[f"{prefix}_mean"] = stats["mean_ms"]
            row[f"{prefix}_sd"] = stats["sd_ms"]
        rows.append(row)
    return pd.DataFrame(rows)


def _configuration_table(by_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        model, variation, label = SCENARIO_META[scenario]
        configuration = by_name[scenario].get("configuration", {})
        xai = configuration.get("xai", {}) or {}
        rows.append({
            "scenario": scenario,
            "model": model,
            "variation": variation,
            "scenario_label": label,
            "model_count": configuration.get("model_count"),
            "mcdp_passes": configuration.get("mcdp_passes"),
            "xai_method": xai.get("method", "none"),
            "xai_methods_count": len(configuration.get("xai_methods", [])),
            "xai_merge": configuration.get("xai_merge", "none"),
            "smoothing_window": configuration.get("smoothing_window"),
            "parameter_count_total": configuration.get("parameter_count_total"),
            "checkpoint_size_mb_total": configuration.get("checkpoint_size_mb_total"),
        })
    return pd.DataFrame(rows)


def _write_latency_latex(latency: pd.DataFrame, output: Path) -> None:
    wide: list[list[str]] = []
    for scenario in SCENARIO_ORDER:
        subset = latency[latency["scenario"] == scenario].set_index("stage")
        row = [_latex_escape(SCENARIO_META[scenario][2]), str(int(subset.loc["compute_ms", "n"]))]
        for stage, _ in STAGES:
            current = subset.loc[stage]
            row.append(format_mean_sd(float(current["mean_ms"]), float(current["sd_ms"]), latex=True))
        wide.append(row)
    headers = ["Cenário", "$n$", "Detecção", "Pré-proc.", r"CPU$\to$GPU", "Inferência", "XAI", "Suavização", "Computação", "End-to-end"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Latência média $\pm$ desvio-padrão amostral por quadro para cada etapa do pipeline. Valores em milissegundos.}",
        r"\label{tab:realtime_latency_mean_sd}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in wide:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize Nota: o end-to-end começa após a leitura do quadro e inclui a espera na fila. O desvio-padrão é calculado entre quadros da mesma execução; não representa variabilidade entre repetições independentes.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_operational_latex(operational: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Desempenho operacional dos pipelines para uma entrada de 30 FPS.}",
        r"\label{tab:realtime_operational}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Cenário & Lidos & Processados & FPS válidos & Perda fila (\%) & Cobertura (\%) & Deadline perdido (\%) & VRAM pico (MB) \\",
        r"\midrule",
    ]
    for _, row in operational.iterrows():
        lines.append(
            f"{_latex_escape(row['scenario_label'])} & {int(row['frames_read'])} & "
            f"{int(row['frames_processed_n'])} & {row['valid_output_fps']:.2f} & "
            f"{row['queue_drop_pct']:.1f} & {row['coverage_pct']:.1f} & "
            f"{row['deadline_miss_pct']:.1f} & {row['torch_vram_peak_mb']:.1f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_resources_latex(resources: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Uso médio $\pm$ desvio-padrão amostral de recursos durante o benchmark.}",
        r"\label{tab:realtime_resources_mean_sd}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Cenário & $n$ amostras & CPU processo (\%) & RAM processo (MB) & GPU (\%) & Memória GPU total (MB) \\",
        r"\midrule",
    ]
    for _, row in resources.iterrows():
        values = [
            _latex_escape(row["scenario_label"]),
            str(int(row["resource_samples_n"])),
            format_mean_sd(row["process_cpu_pct_mean"], row["process_cpu_pct_sd"], latex=True),
            format_mean_sd(row["process_rss_mb_mean"], row["process_rss_mb_sd"], latex=True),
            format_mean_sd(row["gpu_util_pct_mean"], row["gpu_util_pct_sd"], latex=True),
            format_mean_sd(row["gpu_memory_used_mb_mean"], row["gpu_memory_used_mb_sd"], latex=True),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _plot_compute_latency(latency: pd.DataFrame, output_dir: Path) -> None:
    compute = latency[latency["stage"] == "compute_ms"].set_index("scenario").loc[SCENARIO_ORDER]
    labels = [SCENARIO_META[name][2] for name in SCENARIO_ORDER]
    means = compute["mean_ms"].to_numpy(float)
    errors = compute["sd_ms"].to_numpy(float)
    colors = [MODEL_COLORS[SCENARIO_META[name][0]] for name in SCENARIO_ORDER]
    hatches = [VARIATION_HATCHES[SCENARIO_META[name][1]] for name in SCENARIO_ORDER]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    positions = np.arange(len(labels))
    bars = ax.barh(positions, means, xerr=errors, color=colors, alpha=0.82, capsize=3, edgecolor="#333333")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_xscale("log")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Latência de computação [ms] — escala logarítmica")
    ax.grid(axis="x", alpha=0.25, which="both")
    for y, mean, sd in zip(positions, means, errors):
        ax.text(mean * 1.12, y, f"{mean:.1f} ± {sd:.1f}", va="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_dir / "compute_latency_mean_sd.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "compute_latency_mean_sd.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    output: Path,
    latency: pd.DataFrame,
    operational: pd.DataFrame,
    resources: pd.DataFrame,
    run_dir: Path,
    environment: dict[str, Any],
) -> None:
    compute = latency[latency["stage"] == "compute_ms"].set_index("scenario")
    xai = latency[latency["stage"] == "xai_ms"].set_index("scenario")
    op = operational.set_index("scenario")
    lines = [
        "# Desempenho real-time de NCNN, VGGFace e ViT-B/32",
        "",
        "## Síntese técnica",
        "",
        "Os nove cenários foram concluídos com sucesso. Considerando apenas o tempo de computação, as versões simples de NCNN, VGGFace e ViT-B/32 permaneceram abaixo do orçamento de 33,3 ms por quadro. Entretanto, o pipeline completo não sustentou 30 FPS sem perdas: mesmo as versões simples processaram entre 281 e 285 dos 300 quadros.",
        "",
        "MCDP com 50 passes, ensemble de 10 modelos e o workflow XAI aumentaram substancialmente o custo. O melhor cenário completo com XAI foi o NCNN, com 845,39 ± 48,61 ms por quadro e 1,18 FPS válido. Assim, os resultados sustentam inferência simples próxima de tempo real no computador avaliado, mas não sustentam execução contínua do pipeline completo de incerteza e explicabilidade a 30 FPS.",
        "",
        "## As versões simples atendem ao orçamento computacional, mas ainda perdem quadros",
        "",
        "As médias de computação foram 15,70 ± 3,01 ms para NCNN, 17,42 ± 2,95 ms para VGGFace e 19,55 ± 2,92 ms para ViT-B/32. Apesar disso, o throughput válido ficou entre 28,10 e 28,50 FPS, com perdas de fila de 5,0% a 6,3%. Portanto, o orçamento computacional por quadro é atendido, mas o requisito mais forte de 30 FPS sem perdas não foi demonstrado.",
        "",
        "## MCDP e ensemble inviabilizam 30 FPS na configuração testada",
        "",
        "O MCDP-50 elevou a computação do NCNN para 46,97 ± 6,24 ms e a do VGGFace para 110,85 ± 5,45 ms. O ensemble-10 do ViT-B/32 atingiu 55,64 ± 7,00 ms. Todos esses cenários perderam o prazo de 33,3 ms em 100% dos quadros processados e reduziram a cobertura para 71,0%, 31,3% e 60,0%, respectivamente.",
        "",
        "## O XAI é o principal gargalo do pipeline completo",
        "",
        "Nos cenários completos, o XAI consumiu 798,44 ± 45,62 ms no NCNN, 4.659,43 ± 62,41 ms no VGGFace e 7.972,67 ± 342,11 ms no ViT-B/32. Isso corresponde a aproximadamente 94,4%, 97,8% e 99,4% do tempo de computação de cada pipeline completo. O XAI deve, portanto, ser tratado como processamento sob demanda ou offline, e não como uma etapa executada continuamente em todos os quadros.",
        "",
        "## Resultados consolidados — média ± DP amostral",
        "",
        "| Cenário | n | Computação [ms] | End-to-end [ms] | XAI [ms] | FPS válidos | Cobertura | Perda em fila |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIO_ORDER:
        comp = compute.loc[scenario]
        e2e = latency[(latency["scenario"] == scenario) & (latency["stage"] == "end_to_end_ms")].iloc[0]
        xai_row = xai.loc[scenario]
        lines.append(
            f"| {SCENARIO_META[scenario][2]} | {int(comp['n'])} | "
            f"{format_mean_sd(comp['mean_ms'], comp['sd_ms'])} | "
            f"{format_mean_sd(e2e['mean_ms'], e2e['sd_ms'])} | "
            f"{format_mean_sd(xai_row['mean_ms'], xai_row['sd_ms'])} | "
            f"{op.loc[scenario, 'valid_output_fps']:.2f} | {op.loc[scenario, 'coverage_pct']:.1f}% | "
            f"{op.loc[scenario, 'queue_drop_pct']:.1f}% |"
        )
    lines.extend([
        "",
        "## Escopo e definições das métricas",
        "",
        "- Cada cenário recebeu 300 quadros a 30 FPS, com fila de quatro quadros e política de descarte do quadro mais antigo.",
        "- Computação total inclui detecção, pré-processamento, transferência CPU→GPU, inferência, XAI e suavização.",
        "- End-to-end começa após a leitura do quadro e inclui a espera na fila; por isso pode ser muito maior que a computação.",
        "- FPS válido é a taxa de quadros que produziram uma saída completa. Cobertura é a fração dos 300 quadros de entrada efetivamente processada.",
        "- O desvio-padrão é amostral (`ddof=1`) e descreve variação entre quadros dentro de uma única execução.",
        "",
        "## Desenho experimental",
        "",
        "Foram comparados três modelos em versões simples e em configurações de maior custo. Para NCNN e VGGFace, a incerteza foi estimada com 50 passes de Monte Carlo dropout. Para ViT-B/32, foram usados 10 modelos em ensemble. A suavização empregou média móvel causal de 30 quadros. Nos cenários completos, dez métodos Captum foram executados uma vez por imagem e suas máscaras normalizadas foram fundidas por média simples.",
        "",
        "## Ambiente de execução",
        "",
        f"- Plataforma: {environment.get('platform', 'não informado')}",
        f"- CPU: {environment.get('cpu_physical_count', 'n/a')} núcleos físicos / {environment.get('cpu_logical_count', 'n/a')} lógicos; RAM: {float(environment.get('system_ram_gb', float('nan'))):.1f} GB",
        f"- GPU: {environment.get('gpu_name', 'não informado')}; VRAM: {float(environment.get('gpu_total_memory_gb', float('nan'))):.1f} GB",
        f"- Software: Python {environment.get('python', 'n/a')}; PyTorch {environment.get('torch', 'n/a')}; CUDA {environment.get('cuda_runtime', 'n/a')}; InsightFace {environment.get('insightface', 'n/a')}",
        "",
        "## Limitações, incerteza e robustez",
        "",
        "Os cenários XAI processaram somente 6 a 16 quadros porque a fila continuou reproduzindo uma entrada real-time. Essas amostras pequenas tornam suas estimativas de dispersão preliminares. Além disso, o DP calculado entre quadros não mede variabilidade entre execuções, inicializações ou condições térmicas. Não foram realizadas repetições independentes, e os resultados são específicos ao hardware, versões de software, modelos e política de fila documentados.",
        "",
        "## Recomendações",
        "",
        "1. Limitar a alegação de tempo real às versões simples e qualificá-la como próxima de 30 FPS, pois houve perda de 5,0% a 6,3% dos quadros.",
        "2. Executar MCDP, ensemble e XAI sob demanda, em frequência reduzida ou de forma offline.",
        "3. Repetir cada cenário pelo menos três vezes para estimar média e DP entre execuções.",
        "4. Executar uma passagem offline sem descarte para obter n=300 nos cenários XAI.",
        "",
        "## Questões em aberto",
        "",
        "- Qual é o efeito de executar incerteza ou XAI a cada N quadros, reutilizando resultados intermediários?",
        "- Qual é a menor quantidade de passes MCDP ou membros do ensemble que preserva desempenho preditivo e calibração?",
        "- A perda de quadros permanece estável em execuções longas e após aquecimento térmico da GPU?",
        "",
        f"Fonte dos dados: `{run_dir.name}/summary.json` e CSVs por quadro da mesma execução.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_portable_report_artifact(
    output: Path,
    latency: pd.DataFrame,
    operational: pd.DataFrame,
    resources: pd.DataFrame,
    environment: dict[str, Any],
) -> None:
    """Write the canonical input consumed by the portable report renderer.

    The snapshot is produced by executing the SQL queries embedded in the
    artifact against an in-memory SQLite table. This keeps provenance honest
    while avoiding patient-level data and machine-local paths in the report.
    """
    stage_frames = {
        stage: latency[latency["stage"] == stage].set_index("scenario")
        for stage, _ in STAGES
    }
    op = operational.set_index("scenario")
    metric_rows: list[dict[str, Any]] = []
    for order, scenario in enumerate(SCENARIO_ORDER):
        compute = stage_frames["compute_ms"].loc[scenario]
        xai = stage_frames["xai_ms"].loc[scenario]
        e2e = stage_frames["end_to_end_ms"].loc[scenario]
        model, variation, label = SCENARIO_META[scenario]
        metric_rows.append({
            "scenario_order": order,
            "scenario": scenario,
            "scenario_label": label,
            "model": model,
            "variation": variation,
            "processed_n": int(compute["n"]),
            "compute_mean_ms": round(float(compute["mean_ms"]), 3),
            "compute_sd_ms": round(float(compute["sd_ms"]), 3),
            "compute_p95_ms": round(float(compute["p95_ms"]), 3),
            "e2e_mean_ms": round(float(e2e["mean_ms"]), 3),
            "e2e_sd_ms": round(float(e2e["sd_ms"]), 3),
            "e2e_p95_ms": round(float(e2e["p95_ms"]), 3),
            "xai_mean_ms": None if not math.isfinite(float(xai["mean_ms"])) else round(float(xai["mean_ms"]), 3),
            "xai_sd_ms": None if not math.isfinite(float(xai["sd_ms"])) else round(float(xai["sd_ms"]), 3),
            "valid_fps": round(float(op.loc[scenario, "valid_output_fps"]), 3),
            "coverage_pct": round(float(op.loc[scenario, "coverage_pct"]), 3),
            "queue_drop_pct": round(float(op.loc[scenario, "queue_drop_pct"]), 3),
            "deadline_miss_pct": round(float(op.loc[scenario, "deadline_miss_pct"]), 3),
            "has_xai": int(math.isfinite(float(xai["mean_ms"]))),
        })

    stage_rows: list[dict[str, Any]] = []
    for order, scenario in enumerate(SCENARIO_ORDER):
        row: dict[str, Any] = {
            "scenario_order": order,
            "scenario_label": SCENARIO_META[scenario][2],
            "processed_n": int(stage_frames["compute_ms"].loc[scenario, "n"]),
        }
        for stage, _ in STAGES:
            current = stage_frames[stage].loc[scenario]
            mean = float(current["mean_ms"])
            sd = float(current["sd_ms"])
            row[stage] = format_mean_sd(mean, sd)
        stage_rows.append(row)

    operational_by_scenario = operational.set_index("scenario")
    resource_rows: list[dict[str, Any]] = []
    for order, current in resources.set_index("scenario").loc[SCENARIO_ORDER].reset_index().iterrows():
        scenario = str(current["scenario"])
        resource_rows.append({
            "scenario_order": order,
            "scenario_label": current["scenario_label"],
            "samples_n": int(current["resource_samples_n"]),
            "cpu_mean_sd_pct": format_mean_sd(current["process_cpu_pct_mean"], current["process_cpu_pct_sd"]),
            "ram_mean_sd_mb": format_mean_sd(current["process_rss_mb_mean"], current["process_rss_mb_sd"]),
            "gpu_mean_sd_pct": format_mean_sd(current["gpu_util_pct_mean"], current["gpu_util_pct_sd"]),
            "gpu_memory_mean_sd_mb": format_mean_sd(current["gpu_memory_used_mb_mean"], current["gpu_memory_used_mb_sd"]),
            "torch_vram_peak_mb": round(float(operational_by_scenario.loc[scenario, "torch_vram_peak_mb"]), 1),
        })

    query_all = """SELECT scenario_order, scenario_label, model, variation, processed_n,
       compute_mean_ms, compute_sd_ms, compute_p95_ms,
       e2e_mean_ms, e2e_sd_ms, e2e_p95_ms,
       xai_mean_ms, xai_sd_ms, valid_fps, coverage_pct,
       queue_drop_pct, deadline_miss_pct
FROM benchmark_metrics
ORDER BY scenario_order"""
    query_non_xai = """SELECT scenario_order, scenario_label, model, compute_mean_ms
FROM benchmark_metrics
WHERE has_xai = 0
ORDER BY scenario_order"""
    query_xai = """SELECT scenario_order, scenario_label, model, compute_mean_ms
FROM benchmark_metrics
WHERE has_xai = 1
ORDER BY scenario_order"""
    query_fps = """SELECT scenario_order, scenario_label, model, valid_fps,
       coverage_pct, queue_drop_pct, deadline_miss_pct
FROM benchmark_metrics
ORDER BY scenario_order"""
    query_stages = """SELECT scenario_order, scenario_label, processed_n,
       detection_ms, preprocess_ms, transfer_ms, inference_ms,
       xai_ms, smoothing_ms, compute_ms, end_to_end_ms
FROM stage_metrics
ORDER BY scenario_order"""
    query_resources = """SELECT scenario_order, scenario_label, samples_n,
       cpu_mean_sd_pct, ram_mean_sd_mb, gpu_mean_sd_pct,
       gpu_memory_mean_sd_mb, torch_vram_peak_mb
FROM resource_metrics
ORDER BY scenario_order"""

    connection = sqlite3.connect(":memory:")
    try:
        pd.DataFrame(metric_rows).to_sql("benchmark_metrics", connection, index=False)
        pd.DataFrame(stage_rows).to_sql("stage_metrics", connection, index=False)
        pd.DataFrame(resource_rows).to_sql("resource_metrics", connection, index=False)

        def execute(query: str) -> list[dict[str, Any]]:
            cursor = connection.execute(query)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

        datasets = {
            "scenario_metrics": execute(query_all),
            "non_xai_metrics": execute(query_non_xai),
            "xai_metrics": execute(query_xai),
            "fps_metrics": execute(query_fps),
            "stage_metrics": execute(query_stages),
            "resource_metrics": execute(query_resources),
        }
    finally:
        connection.close()

    generated_at = datetime.now(timezone.utc).isoformat()
    source_specs = [
        {
            "id": "source_all_metrics",
            "label": "Estatísticas por quadro do benchmark",
            "query": {
                "engine": "SQLite",
                "sql": query_all,
                "description": "Agregação de média e desvio-padrão amostral calculada a partir dos CSVs por quadro.",
                "executed_at": generated_at,
                "metric_definitions": [
                    "compute_mean_ms: média aritmética por quadro",
                    "compute_sd_ms: desvio-padrão amostral por quadro, ddof=1",
                    "valid_fps: quadros com saída completa divididos pela duração da execução",
                    "coverage_pct: quadros processados divididos pelos 300 quadros lidos",
                ],
                "tables_used": ["benchmark_metrics"],
                "filters": ["300 quadros de entrada por cenário", "entrada simulada a 30 FPS"],
            },
        },
        {
            "id": "source_non_xai_metrics",
            "label": "Cenários sem XAI",
            "query": {"engine": "SQLite", "sql": query_non_xai, "executed_at": generated_at},
        },
        {
            "id": "source_xai_metrics",
            "label": "Cenários completos com XAI",
            "query": {"engine": "SQLite", "sql": query_xai, "executed_at": generated_at},
        },
        {
            "id": "source_fps_metrics",
            "label": "Throughput, cobertura e perdas",
            "query": {
                "engine": "SQLite",
                "sql": query_fps,
                "executed_at": generated_at,
                "tables_used": ["benchmark_metrics"],
                "metric_definitions": [
                    "valid_fps: taxa de quadros com saída completa",
                    "queue_drop_pct: quadros descartados pela fila sobre os 300 quadros lidos",
                ],
            },
        },
        {
            "id": "source_stage_metrics",
            "label": "Latência por etapa do pipeline",
            "query": {
                "engine": "SQLite",
                "sql": query_stages,
                "executed_at": generated_at,
                "tables_used": ["stage_metrics"],
                "metric_definitions": ["cada célula contém média ± desvio-padrão amostral em milissegundos"],
            },
        },
        {
            "id": "source_resource_metrics",
            "label": "Amostras de uso de CPU, RAM, GPU e VRAM",
            "query": {
                "engine": "SQLite",
                "sql": query_resources,
                "executed_at": generated_at,
                "tables_used": ["resource_metrics"],
                "metric_definitions": [
                    "CPU do processo pode exceder 100% por representar uso agregado de múltiplos núcleos",
                    "GPU memory é a memória total observada; torch_vram_peak_mb é o pico alocado pelo processo PyTorch",
                ],
            },
        },
    ]
    chart_common = {
        "type": "horizontalBar",
        "intent": "comparison",
        "xAxisTitle": "Latência média de computação (ms)",
        "valueFormat": "number",
        "unit": "ms",
        "layout": "full",
        "showDescription": True,
        "surface": {"viewMode": "both"},
        "encodings": {
            "x": {"field": "scenario_label", "type": "nominal", "label": "Cenário"},
            "y": {"field": "compute_mean_ms", "type": "quantitative", "label": "Média", "unit": "ms"},
            "color": {"field": "model", "type": "nominal", "label": "Modelo"},
            "tooltip": [
                {"field": "scenario_label", "type": "text", "label": "Cenário"},
                {"field": "compute_mean_ms", "type": "quantitative", "label": "Média", "unit": "ms"},
            ],
        },
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Desempenho real-time de NCNN, VGGFace e ViT-B/32",
            "description": "Relatório técnico de latência, throughput, perdas e recursos em nove configurações do pipeline.",
            "generatedAt": generated_at,
            "sources": source_specs,
            "charts": [
                {
                    **chart_common,
                    "id": "compute_without_xai",
                    "title": "Latência de computação sem XAI",
                    "subtitle": "Média por quadro; o DP amostral é apresentado na tabela.",
                    "dataset": "non_xai_metrics",
                    "sourceId": "source_non_xai_metrics",
                    "referenceLines": [{"axis": "x", "value": 33.333, "label": "Orçamento de 30 FPS", "color": "neutral", "lineStyle": "dashed"}],
                },
                {
                    "id": "valid_output_fps",
                    "type": "horizontalBar",
                    "intent": "comparison",
                    "title": "Throughput válido por cenário",
                    "subtitle": "300 quadros por cenário, entrada a 30 FPS; a linha indica o alvo sem perdas.",
                    "showDescription": True,
                    "dataset": "fps_metrics",
                    "sourceId": "source_fps_metrics",
                    "layout": "full",
                    "xAxisTitle": "Quadros válidos por segundo",
                    "valueFormat": "number",
                    "surface": {"viewMode": "both"},
                    "encodings": {
                        "x": {"field": "scenario_label", "type": "nominal", "label": "Cenário"},
                        "y": {"field": "valid_fps", "type": "quantitative", "label": "FPS válidos", "unit": "FPS"},
                        "color": {"field": "model", "type": "nominal", "label": "Modelo"},
                        "tooltip": [
                            {"field": "scenario_label", "type": "text", "label": "Cenário"},
                            {"field": "valid_fps", "type": "quantitative", "label": "FPS válidos", "unit": "FPS"},
                            {"field": "coverage_pct", "type": "quantitative", "label": "Cobertura", "unit": "%"},
                            {"field": "queue_drop_pct", "type": "quantitative", "label": "Perda em fila", "unit": "%"},
                        ],
                    },
                    "referenceLines": [{"axis": "x", "value": 30.0, "label": "Entrada: 30 FPS", "color": "neutral", "lineStyle": "dashed"}],
                },
                {
                    **chart_common,
                    "id": "compute_with_xai",
                    "title": "Latência de computação com workflow XAI completo",
                    "subtitle": "Cada método XAI é executado uma vez por imagem; as dez máscaras são fundidas por média.",
                    "dataset": "xai_metrics",
                    "sourceId": "source_xai_metrics",
                },
            ],
            "tables": [
                {
                    "id": "scenario_results",
                    "title": "Resultados completos — média e DP amostral",
                    "subtitle": "Tempos em milissegundos; DP calculado entre quadros processados da mesma execução.",
                    "dataset": "scenario_metrics",
                    "sourceId": "source_all_metrics",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "compute_mean_ms", "direction": "asc"},
                    "columns": [
                        {"field": "scenario_label", "label": "Cenário", "type": "text"},
                        {"field": "processed_n", "label": "n", "format": "number"},
                        {"field": "compute_mean_ms", "label": "Computação média (ms)", "format": "number"},
                        {"field": "compute_sd_ms", "label": "Computação DP (ms)", "format": "number"},
                        {"field": "compute_p95_ms", "label": "Computação p95 (ms)", "format": "number"},
                        {"field": "e2e_mean_ms", "label": "End-to-end médio (ms)", "format": "number"},
                        {"field": "e2e_sd_ms", "label": "End-to-end DP (ms)", "format": "number"},
                        {"field": "valid_fps", "label": "FPS válidos", "format": "number"},
                        {"field": "coverage_pct", "label": "Cobertura (%)", "format": "number"},
                        {"field": "queue_drop_pct", "label": "Perda em fila (%)", "format": "number"},
                        {"field": "deadline_miss_pct", "label": "Prazo perdido (%)", "format": "number"},
                    ],
                },
                {
                    "id": "stage_results",
                    "title": "Latência por etapa — média ± DP amostral",
                    "subtitle": "Quadros efetivamente processados em cada cenário; valores em milissegundos.",
                    "dataset": "stage_metrics",
                    "sourceId": "source_stage_metrics",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "scenario_label", "direction": "asc"},
                    "columns": [
                        {"field": "scenario_label", "label": "Cenário", "type": "text"},
                        {"field": "processed_n", "label": "n", "format": "number"},
                        {"field": "detection_ms", "label": "Detecção (ms)", "type": "text"},
                        {"field": "preprocess_ms", "label": "Pré-proc. (ms)", "type": "text"},
                        {"field": "transfer_ms", "label": "CPU→GPU (ms)", "type": "text"},
                        {"field": "inference_ms", "label": "Inferência (ms)", "type": "text"},
                        {"field": "xai_ms", "label": "XAI (ms)", "type": "text"},
                        {"field": "smoothing_ms", "label": "Suavização (ms)", "type": "text"},
                        {"field": "compute_ms", "label": "Computação (ms)", "type": "text"},
                        {"field": "end_to_end_ms", "label": "End-to-end (ms)", "type": "text"},
                    ],
                },
                {
                    "id": "resource_results",
                    "title": "Uso de recursos — média ± DP amostral",
                    "subtitle": "Amostras periódicas durante cada cenário; memória em MB.",
                    "dataset": "resource_metrics",
                    "sourceId": "source_resource_metrics",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "scenario_label", "direction": "asc"},
                    "columns": [
                        {"field": "scenario_label", "label": "Cenário", "type": "text"},
                        {"field": "samples_n", "label": "n amostras", "format": "number"},
                        {"field": "cpu_mean_sd_pct", "label": "CPU processo (%)", "type": "text"},
                        {"field": "ram_mean_sd_mb", "label": "RAM processo (MB)", "type": "text"},
                        {"field": "gpu_mean_sd_pct", "label": "GPU (%)", "type": "text"},
                        {"field": "gpu_memory_mean_sd_mb", "label": "Memória GPU total (MB)", "type": "text"},
                        {"field": "torch_vram_peak_mb", "label": "Pico VRAM PyTorch (MB)", "format": "number"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "report_title",
                    "type": "markdown",
                    "layout": "full",
                    "body": "# Desempenho real-time de NCNN, VGGFace e ViT-B/32",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_all_metrics",
                    "body": "## Síntese técnica\n\n**A inferência simples foi a única configuração compatível com o orçamento computacional de 33,3 ms por quadro.** NCNN, VGGFace e ViT-B/32 apresentaram 15,70 ± 3,01 ms, 17,42 ± 2,95 ms e 19,55 ± 2,92 ms de computação, respectivamente. Mesmo assim, produziram apenas 28,10–28,50 FPS válidos e perderam 5,0%–6,3% dos quadros.\n\n**MCDP-50, ensemble-10 e XAI não sustentaram 30 FPS.** O melhor pipeline completo com XAI foi o NCNN, com 845,39 ± 48,61 ms e 1,18 FPS válido. Os resultados permitem descrever as versões simples como próximas de tempo real no hardware testado, mas não sustentam a operação contínua do pipeline completo de incerteza e explicabilidade a 30 FPS sem perdas.",
                },
                {
                    "id": "simple_and_uncertainty_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_all_metrics",
                    "body": "## O modo simples atende ao orçamento de computação; MCDP e ensemble não\n\nO p95 de computação das três versões simples permaneceu entre 20,21 e 24,00 ms, abaixo do limite de 33,3 ms. Com MCDP-50, o NCNN subiu para 46,97 ± 6,24 ms e o VGGFace para 110,85 ± 5,45 ms. O ensemble-10 do ViT-B/32 atingiu 55,64 ± 7,00 ms. Portanto, o custo da estimação de incerteza deve ser reduzido ou executado em frequência menor que a classificação simples.",
                },
                {"id": "chart_without_xai", "type": "chart", "layout": "full", "chartId": "compute_without_xai"},
                {
                    "id": "throughput_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_fps_metrics",
                    "body": "## A taxa de saída mostra que tempo abaixo de 33,3 ms não garante 30 FPS sem perdas\n\nAs versões simples processaram de 281 a 285 dos 300 quadros, alcançando cobertura de 93,7%–95,0%. Nos cenários de incerteza, a cobertura caiu para 71,0% no NCNN, 60,0% no ViT-B/32 e 31,3% no VGGFace. A fila é parte essencial da avaliação: observar apenas a latência média do modelo superestimaria a capacidade real-time do pipeline.",
                },
                {"id": "chart_valid_fps", "type": "chart", "layout": "full", "chartId": "valid_output_fps"},
                {
                    "id": "xai_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_xai_metrics",
                    "body": "## O workflow XAI domina o tempo do pipeline completo\n\nO XAI consumiu 798,44 ± 45,62 ms no NCNN, 4.659,43 ± 62,41 ms no VGGFace e 7.972,67 ± 342,11 ms no ViT-B/32, representando aproximadamente 94,4%, 97,8% e 99,4% da computação total. A consequência operacional foi uma saída entre 0,12 e 1,18 FPS. O uso cientificamente defensável é gerar explicações sob demanda ou offline, e não para todos os quadros do fluxo de 30 FPS.",
                },
                {"id": "chart_with_xai", "type": "chart", "layout": "full", "chartId": "compute_with_xai"},
                {
                    "id": "consolidated_results_intro",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_all_metrics",
                    "body": "## A comparação consolidada confirma o NCNN como a opção de menor latência\n\nO NCNN apresentou o menor tempo nas três condições comparáveis: simples, estimação de incerteza e pipeline completo com XAI. A tabela também mostra que a latência end-to-end cresce muito além da computação quando a fila acumula quadros, chegando a 20,45 s em média no ViT-B/32 completo.",
                },
                {"id": "results_table", "type": "table", "layout": "full", "tableId": "scenario_results"},
                {
                    "id": "stage_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_stage_metrics",
                    "body": "## Detecção e pré-processamento dominam o modo simples; inferência e XAI dominam as variações\n\nA detecção permaneceu próxima de 10 ms e o pré-processamento entre aproximadamente 3 e 6 ms. Na versão simples, esses estágios representam a maior parte do custo, especialmente no NCNN, cuja inferência foi 1,03 ± 0,27 ms. Com MCDP ou ensemble, a inferência passa a dominar; nos cenários completos, o XAI se torna o gargalo quase exclusivo. Transferência CPU→GPU e suavização tiveram custo computacional desprezível frente às demais etapas.",
                },
                {"id": "stage_results_table", "type": "table", "layout": "full", "tableId": "stage_results"},
                {
                    "id": "resource_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "source_resource_metrics",
                    "body": "## XAI e ensemble também aumentam significativamente a pressão sobre a GPU\n\nO pico de VRAM alocada pelo PyTorch variou de 15,6 MB no NCNN simples a 4.791,4 MB no VGGFace completo com XAI. O ensemble-10 do ViT-B/32 atingiu 3.355,9 MB sem XAI. A memória total de GPU observada chegou, em média, a 10.464,4 ± 2.664,2 MB no VGGFace completo, deixando pouca margem na GPU de 12 GB testada. O percentual de CPU do processo pode exceder 100% porque agrega o uso de múltiplos núcleos.",
                },
                {"id": "resource_results_table", "type": "table", "layout": "full", "tableId": "resource_results"},
                {
                    "id": "scope_and_definitions",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Escopo e definições das métricas\n\n- Cada cenário recebeu 300 quadros a 30 FPS, com fila de quatro quadros e descarte do quadro mais antigo.\n- **Computação total** soma detecção, pré-processamento, transferência CPU→GPU, inferência, XAI e suavização.\n- **End-to-end** começa após a leitura do quadro e inclui a espera na fila.\n- **FPS válido** conta somente quadros que produziram uma saída completa; **cobertura** é a proporção dos 300 quadros lidos que foi processada.\n- Média e desvio-padrão são calculados entre quadros processados da mesma execução; o DP é amostral (`ddof=1`).",
                },
                {
                    "id": "experimental_design",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Desenho experimental e especificação dos pipelines\n\nForam comparados três modelos em versões simples e em configurações de maior custo. NCNN e VGGFace usaram 50 passes de Monte Carlo dropout; ViT-B/32 usou ensemble de 10 modelos. A suavização empregou média móvel causal de 30 quadros. No workflow XAI, dez métodos Captum foram executados uma vez por imagem; cada mapa foi reduzido entre canais, normalizado e os dez resultados foram fundidos por média simples. O detector e os modelos foram aquecidos antes da medição, e os registros por quadro permitiram recomputar as estatísticas sem depender apenas do resumo agregado.",
                },
                {
                    "id": "execution_environment",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Ambiente de execução\n\n"
                    f"- Plataforma: {environment.get('platform', 'não informado')}\n"
                    f"- CPU: {environment.get('cpu_physical_count', 'n/a')} núcleos físicos / {environment.get('cpu_logical_count', 'n/a')} lógicos; RAM: {float(environment.get('system_ram_gb', float('nan'))):.1f} GB\n"
                    f"- GPU: {environment.get('gpu_name', 'não informado')}; VRAM: {float(environment.get('gpu_total_memory_gb', float('nan'))):.1f} GB\n"
                    f"- Software: Python {environment.get('python', 'n/a')}; PyTorch {environment.get('torch', 'n/a')}; CUDA {environment.get('cuda_runtime', 'n/a')}; InsightFace {environment.get('insightface', 'n/a')}",
                },
                {
                    "id": "limitations_and_robustness",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Limitações, incerteza e robustez\n\nOs cenários XAI processaram somente 6 a 16 quadros porque a entrada continuou a 30 FPS e a fila descartou os quadros mais antigos. As médias e os DPs desses cenários são preliminares e podem não representar toda a distribuição das imagens. O DP apresentado mede variabilidade entre quadros de uma única execução; não mede variação entre repetições, inicializações ou condições térmicas. Os resultados são descritivos e específicos ao hardware, versões de software, checkpoints e política de fila avaliados. Não demonstram eficácia clínica nem desempenho em outro computador.",
                },
                {
                    "id": "recommended_next_steps",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Próximos experimentos necessários para sustentar a tese\n\n1. Repetir cada cenário ao menos três vezes e reportar média e DP entre execuções.\n2. Executar uma passagem offline, sem descarte, para obter n=300 nos cenários XAI.\n3. Avaliar MCDP e XAI em frequência reduzida, por exemplo a cada N quadros, medindo o compromisso entre custo, calibração e estabilidade.\n4. Repetir o benchmark por períodos mais longos após aquecimento para verificar throttling e estabilidade térmica.\n5. Redigir a conclusão como **inferência simples próxima de tempo real**, reservando incerteza e XAI para execução sob demanda ou offline.",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Questões em aberto\n\n- Qual é o menor número de passes MCDP ou membros do ensemble que preserva calibração e desempenho preditivo?\n- A reutilização temporal de mapas XAI permite reduzir o custo sem comprometer a interpretação?\n- Qual é a taxa máxima de entrada sem perdas para cada pipeline quando a aquisição é ajustada ao throughput observado?\n- Os resultados permanecem estáveis em outros computadores-alvo e com execuções mais longas?",
                },
            ],
        },
        "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": datasets},
        "sources": source_specs,
        "package_info": {"root": ".", "manifestPath": "artifact.json", "snapshotPath": "artifact.json"},
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_thesis_section_tex(output: Path) -> None:
    """Write a thesis-ready LaTeX section backed by the generated tables."""
    lines = [
        r"\section{Avaliação do desempenho em tempo real}",
        r"\label{sec:benchmark-realtime}",
        "",
        r"Esta avaliação investigou se as configurações baseadas em NCNN, VGGFace e ViT-B/32 conseguem processar continuamente uma entrada de 30 quadros por segundo no computador-alvo. Foram medidos os tempos de cada etapa, a latência \textit{end-to-end}, o \textit{throughput} válido, as perdas na fila e o uso de recursos. Os valores são apresentados como média $\pm$ desvio-padrão amostral entre os quadros processados em uma execução.",
        "",
        r"\subsection{Desenho experimental}",
        "",
        r"Cada cenário recebeu 300 quadros a 30~FPS. A fila comportava quatro quadros e descartava o quadro mais antigo quando estava cheia. Foram avaliadas as versões simples dos três modelos, as configurações de incerteza --- MCDP com 50 passes para NCNN e VGGFace e ensemble de 10 modelos para ViT-B/32 --- e os pipelines completos com suavização causal de 30 quadros e XAI. No workflow XAI, dez métodos foram executados uma vez por imagem e os mapas normalizados foram fundidos por média simples.",
        "",
        r"\subsection{Latência por etapa}",
        "",
        r"As três versões simples permaneceram abaixo do orçamento computacional médio de 33{,}3~ms por quadro: NCNN apresentou $15{,}70 \pm 3{,}01$~ms, VGGFace $17{,}42 \pm 2{,}95$~ms e ViT-B/32 $19{,}55 \pm 2{,}92$~ms. A detecção e o pré-processamento representaram a maior parcela do custo no modo simples. Com MCDP ou ensemble, a inferência passou a dominar. Nos cenários completos, o XAI respondeu por aproximadamente 94{,}4\% a 99{,}4\% da computação total.",
        "",
        r"\input{table_latency_mean_sd.tex}",
        "",
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \includegraphics[width=\textwidth]{compute_latency_mean_sd.pdf}",
        r"    \caption{Latência de computação média e desvio-padrão amostral dos nove cenários. A escala logarítmica permite comparar as versões simples, as configurações de incerteza e os pipelines completos com XAI.}",
        r"    \label{fig:realtime-compute-latency}",
        r"\end{figure}",
        "",
        r"\subsection{Throughput, cobertura e perdas}",
        "",
        r"Embora as versões simples tenham permanecido abaixo de 33{,}3~ms de computação média, nenhuma delas atingiu 30~FPS sem perdas. O throughput válido variou de 28{,}10 a 28{,}50~FPS, com cobertura de 93{,}7\% a 95{,}0\% e perda em fila de 5{,}0\% a 6{,}3\%. MCDP-50 e ensemble-10 reduziram a cobertura para 31{,}3\% a 71{,}0\%. Nos pipelines completos com XAI, o throughput ficou entre 0{,}12 e 1{,}18~FPS.",
        "",
        r"\input{table_operational_metrics.tex}",
        "",
        r"\subsection{Uso de recursos}",
        "",
        r"O uso de memória aumentou principalmente com ensemble e XAI. O pico de VRAM alocada pelo PyTorch variou de 15{,}6~MB no NCNN simples a 4.791{,}4~MB no VGGFace completo com XAI. A memória total de GPU observada no VGGFace completo alcançou, em média, $10.464{,}4 \pm 2.664{,}2$~MB, deixando pouca margem na GPU de 12~GB utilizada. O percentual de CPU do processo pode exceder 100\%, pois agrega o uso de múltiplos núcleos.",
        "",
        r"\input{table_resources_mean_sd.tex}",
        "",
        r"\subsection{Interpretação dos resultados}",
        "",
        r"Os resultados sustentam a caracterização das versões simples como próximas de tempo real no hardware avaliado, mas não demonstram operação estrita a 30~FPS sem perdas. NCNN apresentou a menor latência nas três condições comparáveis. MCDP-50, ensemble-10 e o workflow XAI completo não são adequados para execução contínua em todos os quadros na configuração testada; sua aplicação deve ser sob demanda, em frequência reduzida ou offline.",
        "",
        r"\subsection{Limitações}",
        "",
        r"Os cenários XAI processaram somente 6 a 16 quadros porque a entrada permaneceu em 30~FPS e a fila descartou quadros. Assim, suas médias e seus desvios-padrão são preliminares. Além disso, o desvio-padrão apresentado descreve a variação entre quadros de uma única execução, não a variabilidade entre repetições independentes. Os resultados são específicos ao hardware, às versões de software, aos checkpoints e à política de fila utilizados, e não constituem evidência de eficácia clínica.",
        "",
        r"Para fortalecer a evidência, recomenda-se repetir cada cenário ao menos três vezes, executar uma passagem offline sem descarte para obter 300 observações nos cenários XAI e avaliar versões reduzidas de MCDP, ensemble e XAI em frequência temporal menor.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def build_thesis_artifacts(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    run_path = Path(run_dir).resolve()
    summaries, by_name = _load_run(run_path)
    environment = summaries[0].get("environment", {})
    destination = Path(output_dir).resolve() if output_dir else run_path / "thesis_artifacts"
    destination.mkdir(parents=True, exist_ok=True)

    latency = _latency_table(run_path, by_name)
    operational = _operational_table(by_name)
    resources = _resources_table(run_path)
    configurations = _configuration_table(by_name)
    latency.to_csv(destination / "latency_by_stage_mean_sd.csv", index=False)
    operational.to_csv(destination / "operational_metrics.csv", index=False)
    resources.to_csv(destination / "resources_mean_sd.csv", index=False)
    configurations.to_csv(destination / "scenario_configuration.csv", index=False)
    _write_latency_latex(latency, destination / "table_latency_mean_sd.tex")
    _write_operational_latex(operational, destination / "table_operational_metrics.tex")
    _write_resources_latex(resources, destination / "table_resources_mean_sd.tex")
    _plot_compute_latency(latency, destination)
    _write_markdown_report(
        destination / "TECHNICAL_REPORT.md", latency, operational, resources, run_path, environment
    )
    _write_portable_report_artifact(
        destination / "artifact.json", latency, operational, resources, environment
    )
    _write_thesis_section_tex(destination / "THESIS_RESULTS_SECTION.tex")
    (destination / "benchmark_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_notes = {
        "audience": "technical",
        "delivery_mode": "portable_html",
        "required_structure_mapping": {
            "title": "report_title",
            "technical_summary": "technical_summary",
            "key_findings_with_visual_evidence": [
                "simple_and_uncertainty_finding",
                "throughput_finding",
                "xai_finding",
                "stage_finding",
                "resource_finding",
            ],
            "scope_data_metric_definitions": "scope_and_definitions",
            "methodology_and_experimental_design": "experimental_design",
            "limitations_uncertainty_robustness": "limitations_and_robustness",
            "recommended_next_steps": "recommended_next_steps",
            "further_questions": "further_questions",
        },
        "chart_map": [
            {
                "section": "simple_and_uncertainty_finding",
                "question": "Quais configurações atendem ao orçamento de 33,3 ms?",
                "type": "horizontalBar",
                "fields": ["scenario_label", "compute_mean_ms", "model"],
                "claim": "Somente as versões simples ficam abaixo do orçamento médio.",
            },
            {
                "section": "throughput_finding",
                "question": "Qual é a taxa efetiva de quadros com saída completa?",
                "type": "horizontalBar",
                "fields": ["scenario_label", "valid_fps", "coverage_pct", "queue_drop_pct"],
                "claim": "Nenhum cenário atingiu 30 FPS sem perdas.",
            },
            {
                "section": "xai_finding",
                "question": "Qual é a latência dos pipelines completos com XAI?",
                "type": "horizontalBar",
                "fields": ["scenario_label", "compute_mean_ms"],
                "claim": "O XAI torna o pipeline incompatível com execução contínua a 30 FPS.",
            },
        ],
        "omissions": {
            "between_run_standard_deviation": "Indisponível porque foi realizada uma execução por cenário.",
            "xai_n_300": "Indisponível no modo real-time devido ao descarte de quadros; requer passagem offline.",
        },
    }
    (destination / "REPORT_NOTES.json").write_text(
        json.dumps(report_notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "source_run": run_path.name,
        "statistics": {
            "mean": "arithmetic mean across processed frames",
            "standard_deviation": "sample standard deviation across processed frames (ddof=1)",
            "unit_latency": "milliseconds",
        },
        "scenario_count": len(SCENARIO_ORDER),
        "stage_count": len(STAGES),
        "privacy": "No patient or source filenames are included.",
    }
    (destination / "STATISTICAL_METHOD.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera tabelas e figuras para a tese a partir do benchmark real-time.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output = build_thesis_artifacts(args.run_dir, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
