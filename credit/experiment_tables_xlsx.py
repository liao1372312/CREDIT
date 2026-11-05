"""Export the audited CREDIT experiment tables to one Excel sheet per table."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries


SHEETS = (
    "T1_Public_Main",
    "T2_Attribution",
    "T3_Replay",
    "T4_Certification",
    "T5_Budget",
    "T6_Amortization",
    "T7_Reproducibility",
    "T8_Ablation",
)

CAPTIONS = {
    "T1_Public_Main": "Complete-method evaluation on held-out public ToolSandbox scenarios",
    "T2_Attribution": "Attribution accuracy at matched estimator budgets",
    "T3_Replay": "Replay consistency and physical execution cost",
    "T4_Certification": "Sequential certification under known simulation truth",
    "T5_Budget": "Quality and physical execution cost under matched adaptation budget caps",
    "T6_Amortization": "Cost amortization under declared deployment horizons",
    "T7_Reproducibility": "Frozen reproducibility settings for the additional evaluation",
    "T8_Ablation": "Mechanism ablation with linked controlled diagnostic evidence",
}

NOTES = {
    "T1_Public_Main": (
        "Native score is primary. Full CREDIT deployed the Semantic fallback because the four "
        "certification tasks had no applicable-memory contrast."
    ),
    "T2_Attribution": (
        "The same 180 frozen fixtures are used at every budget. N/A denotes an unsupported "
        "estimand; it is never converted to zero."
    ),
    "T3_Replay": (
        "Scoped replay differed from full replay on 16/20 outcomes under the same-logical "
        "comparison and therefore failed logical equivalence."
    ),
    "T4_Certification": (
        "Each condition uses 2,000 independent streams, eight strata, 200 labeled pairs per "
        "stratum-cycle, and at most 50 cycles."
    ),
    "T5_Budget": (
        "B0=768 physical calls. Unsupported Full CREDIT certification deploys the Semantic "
        "fallback."
    ),
    "T6_Amortization": (
        "Horizons beyond the observed run are stationary projections and do not increase the "
        "experimental sample size."
    ),
    "T7_Reproducibility": "Values are generated from the frozen protocol and audit manifests.",
    "T8_Ablation": (
        "The public ablation panel remains unchanged. The last three diagnostics are planted "
        "controlled stress tests on 32 new seeds, not public-benchmark evidence."
    ),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def na(value: Any) -> Any:
    return "N/A" if value is None else value


def pair(values: list[float] | None) -> tuple[Any, Any]:
    return ("N/A", "N/A") if values is None else (values[0], values[1])


def public_main(root: Path) -> tuple[list[str], list[list[Any]]]:
    main = read(root / "public/main_summary.json")
    contrasts = read(root / "public/primary_contrasts.json")
    groups = {
        "NoMem": "Memory controls",
        "Semantic": "Memory controls",
        "Behavior-filter": "Memory controls",
        "CMP-adapted": "Adaptations of published methods",
        "CAR-adapted": "Adaptations of published methods",
        "Direct-Sandbox": "Execution-based selection",
        "Full CREDIT": "Proposed method",
    }
    headers = [
        "Panel", "Group", "Item", "Sessions", "Native Score", "VSR Rate",
        "Delta Native", "CI Lower", "CI Upper", "Raw p", "Holm p",
        "Adaptation Calls", "Online Calls", "Total Calls",
    ]
    rows = []
    for row in main:
        lo, hi = pair(row["delta_95ci"])
        rows.append([
            "Main methods", groups[row["method"]], row["method"], row["sessions"],
            row["native_score"], row["vsr"], row["delta_vs_semantic"], lo, hi,
            "N/A", "N/A", row["adaptation_calls"], row["online_calls"], row["total_calls"],
        ])
    for baseline in ("Semantic", "Behavior-filter", "Direct-Sandbox"):
        row = contrasts[baseline]
        lo, hi = pair(row["ci"])
        rows.append([
            "Primary contrasts", "Holm family", f"CREDIT versus {baseline}", "N/A",
            "N/A", "N/A", row["delta"], lo, hi, row["raw_p"], row["holm_p"],
            "N/A", "N/A", "N/A",
        ])
    return headers, rows


def attribution(root: Path) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Budget", "Estimator", "Families", "Banzhaf RMSE", "Shapley RMSE",
        "Pair F1", "Triple MAE", "Coverage Rate", "Mean Calls per Family",
        "Total Estimator Calls", "Reference Calls",
    ]
    rows = []
    for row in read(root / "offline/attribution_summary.json"):
        rows.append([
            row["budget"], row["method"], row["families"], na(row["banzhaf_rmse"]),
            na(row["shapley_rmse"]), na(row["pair_f1"]), na(row["triple_mae"]),
            na(row["coverage"]), row["actual_calls_per_family_mean"],
            row["actual_calls_total"], row["reference_calls"],
        ])
    return headers, rows


def replay(root: Path) -> tuple[list[str], list[list[Any]]]:
    value = read(root / "replay/replay_summary.json")
    headers = [
        "Panel", "Item", "Condition", "N", "State Differences", "Trace Differences",
        "Outcome Differences", "Decision Differences", "Calls A", "Calls B",
        "Primary Score", "Accepted Updates", "Unsafe Accepted", "Accepted Denominator",
        "Total Calls", "Measured p95 Seconds",
    ]
    rows = []
    for row in value["comparisons"]:
        rows.append([
            "Consistency", row["comparison"], row["condition"], row["N"],
            row["state_diff"], row["trace_diff"], row["outcome_diff"],
            row["decision_diff"], row["calls_a"], row["calls_b"],
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
        ])
    for row in value["policies"]:
        rows.append([
            "Matched budget", row["policy"], "Same physical budget", "N/A",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", row["primary_score"],
            row["accepted_updates"], row["unsafe_accepted"],
            row["accepted_denominator"], row["total_calls"], row["measured_p95_seconds"],
        ])
    return headers, rows


def certification(root: Path) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Panel", "Truth Condition", "Rule", "Effect Size pp", "False Improvement Rate",
        "Guard Error Rate", "Any-cycle Error Rate", "Error CI Lower", "Error CI Upper",
        "Power Rate", "Power CI Lower", "Power CI Upper", "Certification Calls per Stream",
    ]
    descriptions = {
        "A": "Aggregate null with zero true improvement",
        "B": "Violating stratum despite positive aggregate gain",
        "C": "Beneficial alternatives with every guard satisfied",
    }
    rows = []
    for rule in read(root / "offline/certification_summary.json"):
        for panel_key, panel in (("A", rule["panel_a"]), ("B", rule["panel_b"])):
            lo, hi = pair(panel["any_cycle_error_95ci"])
            rows.append([
                panel_key, descriptions[panel_key], rule["rule"], "N/A",
                na(panel["false_improvement"]), na(panel["guard_error"]),
                panel["any_cycle_error"], lo, hi, "N/A", "N/A", "N/A",
                panel["cert_calls_per_stream"],
            ])
        panel = rule["panel_c"]
        for index, effect in enumerate((2, 5, 10)):
            power_lo, power_hi = pair(panel["power_95ci"][index])
            rows.append([
                "C", descriptions["C"], rule["rule"], effect, "N/A", "N/A", "N/A",
                "N/A", "N/A", panel["power"][index], power_lo, power_hi,
                panel["cert_calls_per_stream"][index],
            ])
    return headers, rows


def budget(root: Path) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Budget Multiplier", "Budget Cap Calls", "Method", "Primary Score",
        "CI Lower", "CI Upper", "Adaptation Calls", "Online Calls", "Total Calls",
        "Exhausted Runs", "Runs",
    ]
    rows = []
    for row in read(root / "public/budget_summary.json"):
        lo, hi = pair(row["score_95ci"])
        rows.append([
            row["multiplier"], int(768 * row["multiplier"]), row["method"],
            row["primary_score"], lo, hi, row["adaptation_calls"], row["online_calls"],
            row["total_calls"], row["exhausted_runs"], row["runs"],
        ])
    return headers, rows


def amortization(root: Path) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Deployment Horizon", "Method", "Initial Cost", "Recurring Update Cost",
        "Cost per Task", "Cost per Verified Success", "Relative Cost vs Semantic",
    ]
    rows = []
    for row in read(root / "public/amortization_summary.json"):
        rows.append([
            row["horizon"], row["method"], row["initial_cost"],
            row["recurring_update_cost"], row["cost_per_task"],
            na(row["cost_per_verified_success"]), na(row["relative_vs_semantic"]),
        ])
    return headers, rows


def reproducibility(root: Path) -> tuple[list[str], list[list[Any]]]:
    public = read(root / "public/frozen_protocol.json")
    offline = read(root / "offline/frozen_protocol.json")
    replay_protocol = read(root / "replay/frozen_protocol.json")
    agent = read(root / "public/test_seed_4103/agent_manifest.json")
    cert_decision = read(root / "public/certification_decision.json")
    attr = offline["attribution"]
    cert = offline["certification"]
    rows = [
        ["Public benchmark", "Repository revision", public["upstream_commit"]],
        ["Public benchmark", "Eligible scenario manifest", f"53 eligible; {len(public['external_dependencies'])} excluded before outcomes"],
        ["Public benchmark", "Development / certification / test families", f"{len(public['development_families'])} / {len(public['certification_families'])} / {len(public['test_families'])}"],
        ["Public benchmark", "External service dependencies", "None in selected tasks; external-search scenarios excluded"],
        ["Actor and executor", "Model snapshot and decoding", f"{agent['model']}; temperature {public['temperature']}; test seeds {public['actor_seeds']['test']}"],
        ["Actor and executor", "Executor revision and prompts", "Frozen argument-aware v3 executor; source hashes archived"],
        ["Actor and executor", "Paired evaluation replications", len(public["actor_seeds"]["test"])],
        ["Actor and executor", "Supported randomness controls", f"Provider seed plus balanced task schedule; exact prompt cache={public['exact_prompt_cache']}"],
        ["Memory and transport", "History source and pool size", f"{public['memory_pool_size']} verified single-tool public recipes; SHA256 {public['memory_pool_sha256']}"],
        ["Memory and transport", "Retriever and encoder snapshot", "Deterministic lexical/tool-role scoring; no learned encoder"],
        ["Memory and transport", "Role construction and manual effort", "Tool-name roles from public schemas; no test-outcome labels"],
        ["Memory and transport", "Drift probes and fixed thresholds", "Pinned upstream; no natural drift opportunity in the public suite"],
        ["Attribution", "Family fixtures", attr["families"]],
        ["Attribution", "Memory count K", ", ".join(map(str, attr["k"]))],
        ["Attribution", "Physical estimator budgets", " / ".join(map(str, attr["budgets"])) + " calls per family"],
        ["Attribution", "Estimands and reference precision", attr["reference"]],
        ["Bank selection", "Numeric budget B0 and unit", f"{public['adaptation_budget_B0']} {public['budget_unit']}"],
        ["Bank selection", "Candidate count and screening rule", "Four frozen exposure arms; native-score development screen"],
        ["Bank selection", "Budget exhaustion fallback", cert_decision["fallback"]],
        ["Certification simulation", "Independent streams per condition", cert["streams_per_condition"]],
        ["Certification simulation", "Strata / labeled pairs per stratum-cycle", f"{cert['strata']} / {cert['labeled_pairs_per_stratum_per_cycle']}"],
        ["Certification simulation", "Maximum cycles / total alpha", f"{cert['maximum_cycles']} / {cert['alpha_star']}"],
        ["Certification simulation", "Margin m / alternative gains", f"{cert['margin']} / 2 pp, 5 pp, 10 pp"],
        ["Certification simulation", "Calibration and unlabeled sizes", f"{cert['labeled_pairs_per_stratum_per_cycle']} labeled pairs / {cert['unlabeled_predictions_per_stratum_per_cycle']:,} unlabeled predictions per stratum-cycle"],
        ["Certification simulation", "Predictor explained variance", cert["predictor_explained_variance"]],
        ["Certification simulation", "Allocation across cycles and guards", cert["alpha_allocation"]],
        ["Replay audit", "Checkpoint and exposure manifests", f"{replay_protocol['population']}; actor seed {replay_protocol['source_actor_seed']}"],
        ["Replay audit", "State and trace normalization", f"{replay_protocol['state_hash']}; {replay_protocol['trace_normalization']}"],
    ]
    return ["Component", "Parameter", "Value / Specification"], rows


def ablation(root: Path) -> tuple[list[str], list[list[Any]]]:
    public_rows = read(root / "public/ablation_summary.json")
    attribution_diag = read(root / "public/mechanism_diagnostics.json")
    support = read(root / "mechanism_support/summary.json")
    headers = [
        "Panel", "Item", "Challenge", "Removed Mechanism", "Endpoint", "Native Score",
        "Delta vs Full", "CI Lower", "CI Upper", "Total Calls", "Certification Status",
        "Full Metric", "Ablated Metric", "Ablated minus Full", "Effect CI Lower",
        "Effect CI Upper", "Full Numerator", "Full Denominator", "Ablated Numerator",
        "Ablated Denominator", "Full Zero-count CP95 Upper", "Raw p", "Holm p",
        "Secondary Result",
    ]
    rows = []
    for row in public_rows:
        lo, hi = pair(row["delta_95ci"])
        rows.append([
            "Public benchmark ablation", row["variant"], "N/A", "N/A",
            "Official native similarity", row["native_score"], row["delta_vs_full"],
            lo, hi, row["total_calls"], row["certification_status"], "N/A", "N/A",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
            "N/A", "N/A", (
                f"Raw uncertified native={row['raw_uncertified_native_score']:.6f}"
                if "raw_uncertified_native_score" in row else "N/A"
            ),
        ])
    lo, hi = attribution_diag["paired_95ci"]
    rows.append([
        "Controlled diagnostic", "Randomized exposure", "Ambiguous Memory Attribution",
        "Randomized exposure", attribution_diag["endpoint"], "N/A", "N/A", "N/A",
        "N/A", "N/A", "controlled fixture", attribution_diag["factorial_aggregate_rmse"],
        attribution_diag["uniform_aggregate_rmse"], attribution_diag["paired_mean_effect"],
        lo, hi, "N/A", attribution_diag["families"], "N/A", attribution_diag["families"],
        "N/A", "N/A", "N/A", f"Budget={attribution_diag['budget']} calls/family",
    ])
    role = support["role_transport"]
    lo, hi = role["paired_95ci"]
    rows.append([
        "Controlled diagnostic", "Role transport", "Uncertain Memory Transfer",
        "Role transport", "Held-out utility prediction MAE", "N/A", "N/A", "N/A",
        "N/A", "N/A", "controlled planted stress", role["structured_mae"],
        role["pooled_mae"], role["pooled_minus_structured_mae"], lo, hi, "N/A",
        role["total_compositions"], "N/A", role["total_compositions"], "N/A",
        role["raw_p"], role["holm_p"], f"32 seeds x {role['compositions_per_seed']} compositions",
    ])
    drift = support["drift_gate"]
    rows.append([
        "Controlled diagnostic", "Drift gate", "Uncertain Memory Transfer", "Drift gate",
        "Unsafe accepted transfers / divergent opportunities", "N/A", "N/A", "N/A",
        "N/A", "N/A", "controlled planted stress", 0.0, 1.0,
        drift["unsafe_rate_increase_without_gate"], drift["unsafe_rate_paired_95ci"][0],
        drift["unsafe_rate_paired_95ci"][1], drift["full_unsafe_accepted"],
        drift["divergent_opportunities"], drift["no_gate_unsafe_accepted"],
        drift["divergent_opportunities"], drift["full_zero_count_cp95"][1],
        drift["raw_p"], drift["holm_p"],
        f"VSR full={drift['full_vsr']:.6f}; no gate={drift['no_gate_vsr']:.6f}; stable false blocks={drift['stable_false_blocks']}/{drift['stable_opportunities']}",
    ])
    cert = support["fresh_certification"]
    full_rate = cert["full_unsafe_accepted"] / cert["full_accepted"]
    ablated_rate = cert["no_certification_unsafe_accepted"] / cert["no_certification_accepted"]
    rows.append([
        "Controlled diagnostic", "Fresh certification", "Budget-Constrained Update Reliability",
        "Fresh certification", "Unsafe accepted updates / accepted updates", "N/A", "N/A",
        "N/A", "N/A", "N/A", "controlled planted stress", full_rate, ablated_rate,
        cert["unsafe_rate_increase_without_certification"], cert["paired_95ci"][0],
        cert["paired_95ci"][1], cert["full_unsafe_accepted"], cert["full_accepted"],
        cert["no_certification_unsafe_accepted"], cert["no_certification_accepted"],
        cert["full_zero_count_cp95"][1], cert["raw_p"], cert["holm_p"],
        f"{support['independent_seeds']} independent seeds; {cert['proposals_per_seed']} proposals/seed",
    ])
    return headers, rows


BUILDERS = {
    "T1_Public_Main": public_main,
    "T2_Attribution": attribution,
    "T3_Replay": replay,
    "T4_Certification": certification,
    "T5_Budget": budget,
    "T6_Amortization": amortization,
    "T7_Reproducibility": reproducibility,
    "T8_Ablation": ablation,
}


def write_sheet(ws, headers: list[str], rows: list[list[Any]], table_index: int) -> None:
    max_col = len(headers)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    ws["A1"] = CAPTIONS[ws.title]
    ws["A1"].font = Font(size=14, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 24

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max_col)
    ws["A2"] = "Source: results/credit_experiment_tables_v1 audited JSON artifacts"
    ws["A2"].font = Font(size=9, italic=True, color="666666")
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=max_col)
    ws["A3"] = NOTES[ws.title]
    ws["A3"].font = Font(size=9, color="444444")
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[3].height = 30

    header_row = 5
    for col, value in enumerate(headers, 1):
        cell = ws.cell(header_row, col, value)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5B9BD5")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[header_row].height = 32

    for row_index, values in enumerate(rows, header_row + 1):
        if len(values) != max_col:
            raise ValueError(f"{ws.title}: row width {len(values)} != {max_col}")
        for col, value in enumerate(values, 1):
            cell = ws.cell(row_index, col, value)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if isinstance(value, float):
                cell.number_format = "0.0000"
        if any(str(value) == "Full CREDIT" for value in values):
            for col in range(1, max_col + 1):
                ws.cell(row_index, col).fill = PatternFill("solid", fgColor="E8F2F9")
                ws.cell(row_index, col).font = Font(bold=True)

    end_row = header_row + len(rows)
    table = Table(displayName=f"CREDITTable{table_index}", ref=f"A{header_row}:{get_column_letter(max_col)}{end_row}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
        showRowStripes=True, showColumnStripes=False,
    )
    ws.add_table(table)
    ws.freeze_panes = "A6"
    ws.auto_filter.ref = table.ref
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "$5:$5"
    ws.auto_filter.ref = table.ref

    thin = Side(style="thin", color="D9E2F3")
    for row in ws.iter_rows(min_row=header_row, max_row=end_row, min_col=1, max_col=max_col):
        for cell in row:
            cell.border = Border(bottom=thin)

    for col, header in enumerate(headers, 1):
        values = [str(header)] + [str(row[col - 1]) for row in rows]
        width = min(42, max(10, max(len(value) for value in values) + 2))
        ws.column_dimensions[get_column_letter(col)].width = width

    percent_headers = {
        "VSR Rate", "Coverage Rate", "False Improvement Rate", "Guard Error Rate",
        "Any-cycle Error Rate", "Error CI Lower", "Error CI Upper", "Power Rate",
        "Power CI Lower", "Power CI Upper", "Full Zero-count CP95 Upper",
    }
    p_headers = {"Raw p", "Holm p"}
    for col, header in enumerate(headers, 1):
        if header in percent_headers:
            for row in range(header_row + 1, end_row + 1):
                if isinstance(ws.cell(row, col).value, (int, float)):
                    ws.cell(row, col).number_format = "0.00%"
        elif header in p_headers:
            for row in range(header_row + 1, end_row + 1):
                if isinstance(ws.cell(row, col).value, (int, float)):
                    ws.cell(row, col).number_format = "0.0000E+00"

    if ws.title == "T8_Ablation":
        item_col = headers.index("Item") + 1
        rate_headers = (
            "Full Metric", "Ablated Metric", "Ablated minus Full",
            "Effect CI Lower", "Effect CI Upper", "Full Zero-count CP95 Upper",
        )
        rate_columns = [headers.index(header) + 1 for header in rate_headers]
        for row in range(header_row + 1, end_row + 1):
            if ws.cell(row, item_col).value in {"Drift gate", "Fresh certification"}:
                for col in rate_columns:
                    if isinstance(ws.cell(row, col).value, (int, float)):
                        ws.cell(row, col).number_format = "0.00%"

    # Highlight the proposed method by formula without changing numeric values.
    if "Item" in headers:
        item_col = get_column_letter(headers.index("Item") + 1)
        ws.conditional_formatting.add(
            f"A{header_row + 1}:{get_column_letter(max_col)}{end_row}",
            FormulaRule(formula=[f'${item_col}{header_row + 1}="Full CREDIT"'],
                        fill=PatternFill("solid", fgColor="E8F2F9")),
        )


def workbook_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


# The LaTeX source contains several multi-panel tables whose panels have
# different schemas.  The first exporter flattened those panels into one wide
# rectangular range and filled irrelevant columns with N/A.  The clean export
# below keeps each source table on one worksheet, but renders each panel as its
# own compact block.  N/A is therefore reserved for a genuinely unsupported
# estimator target in the attribution experiment.


Block = tuple[str, list[str], list[list[Any]]]


def clean_public_main(root: Path) -> list[Block]:
    main = read(root / "public/main_summary.json")
    contrasts = read(root / "public/primary_contrasts.json")
    groups = {
        "NoMem": "Memory controls",
        "Semantic": "Memory controls",
        "Behavior-filter": "Memory controls",
        "CMP-adapted": "Adaptations of published methods",
        "CAR-adapted": "Adaptations of published methods",
        "Direct-Sandbox": "Execution-based selection",
        "Full CREDIT": "Proposed method",
    }
    method_rows = []
    for row in main:
        lo, hi = row["delta_95ci"]
        method_rows.append([
            groups[row["method"]], row["method"], row["sessions"], row["native_score"],
            row["vsr"], row["delta_vs_semantic"], lo, hi, row["adaptation_calls"],
            row["online_calls"], row["total_calls"],
        ])
    contrast_rows = []
    for baseline in ("Semantic", "Behavior-filter", "Direct-Sandbox"):
        row = contrasts[baseline]
        lo, hi = row["ci"]
        contrast_rows.append([
            f"CREDIT versus {baseline}", row["delta"], lo, hi,
            row["raw_p"], row["holm_p"],
        ])
    return [
        (
            "A. Complete methods",
            ["Group", "Method", "Sessions", "Native Score", "VSR Rate", "Delta Native",
             "CI Lower", "CI Upper", "Adaptation Calls", "Online Calls", "Total Calls"],
            method_rows,
        ),
        (
            "B. Prespecified primary contrasts",
            ["Contrast", "Delta Native", "CI Lower", "CI Upper", "Raw p", "Holm p"],
            contrast_rows,
        ),
    ]


def clean_attribution(root: Path) -> list[Block]:
    headers, rows = attribution(root)
    return [("Matched estimator budgets", headers, rows)]


def clean_replay(root: Path) -> list[Block]:
    value = read(root / "replay/replay_summary.json")
    consistency = [[
        row["comparison"], row["condition"], row["N"], row["state_diff"],
        row["trace_diff"], row["outcome_diff"], row["decision_diff"],
        row["calls_a"], row["calls_b"],
    ] for row in value["comparisons"]]
    policies = []
    for row in value["policies"]:
        unsafe_ratio = (
            f"{row['unsafe_accepted']}/{row['accepted_denominator']}"
            if row["accepted_denominator"]
            else "Undefined (0 accepted updates)"
        )
        policies.append([
            row["policy"], row["primary_score"], row["accepted_updates"], unsafe_ratio,
            row["total_calls"], row["measured_p95_seconds"],
        ])
    return [
        (
            "A. Replay consistency",
            ["Comparison", "Fixed Condition", "N", "State Differences", "Trace Differences",
             "Outcome Differences", "Decision Differences", "Calls A", "Calls B"],
            consistency,
        ),
        (
            "B. Performance at the same physical budget",
            ["Policy", "Primary Score", "Accepted Updates", "Unsafe / Accepted Updates",
             "Total Calls", "Measured p95 Seconds"],
            policies,
        ),
    ]


def clean_certification(root: Path) -> list[Block]:
    values = read(root / "offline/certification_summary.json")
    panel_a, panel_b, panel_c = [], [], []
    for rule in values:
        a = rule["panel_a"]
        alo, ahi = a["any_cycle_error_95ci"]
        panel_a.append([
            rule["rule"], a["false_improvement"], a["any_cycle_error"], alo, ahi,
            a["cert_calls_per_stream"],
        ])
        b = rule["panel_b"]
        blo, bhi = b["any_cycle_error_95ci"]
        panel_b.append([
            rule["rule"], b["guard_error"], b["any_cycle_error"], blo, bhi,
            b["cert_calls_per_stream"],
        ])
        c = rule["panel_c"]
        panel_c.append([
            rule["rule"],
            c["power"][0], c["power_95ci"][0][0], c["power_95ci"][0][1], c["cert_calls_per_stream"][0],
            c["power"][1], c["power_95ci"][1][0], c["power_95ci"][1][1], c["cert_calls_per_stream"][1],
            c["power"][2], c["power_95ci"][2][0], c["power_95ci"][2][1], c["cert_calls_per_stream"][2],
        ])
    return [
        (
            "A. Aggregate null with zero true improvement",
            ["Rule", "False Improvement Rate", "Any-cycle Error Rate", "Error CI Lower",
             "Error CI Upper", "Certification Calls per Stream"],
            panel_a,
        ),
        (
            "B. Violating stratum despite positive aggregate gain",
            ["Rule", "Guard Error Rate", "Any-cycle Error Rate", "Error CI Lower",
             "Error CI Upper", "Certification Calls per Stream"],
            panel_b,
        ),
        (
            "C. Beneficial alternatives with every guard satisfied",
            ["Rule", "Power 2 pp", "2 pp CI Lower", "2 pp CI Upper", "Calls at 2 pp",
             "Power 5 pp", "5 pp CI Lower", "5 pp CI Upper", "Calls at 5 pp",
             "Power 10 pp", "10 pp CI Lower", "10 pp CI Upper", "Calls at 10 pp"],
            panel_c,
        ),
    ]


def clean_budget(root: Path) -> list[Block]:
    headers, rows = budget(root)
    return [("Matched adaptation budget caps", headers, rows)]


def clean_amortization(root: Path) -> list[Block]:
    headers, rows = amortization(root)
    return [("Declared deployment horizons", headers, rows)]


def clean_reproducibility(root: Path) -> list[Block]:
    headers, rows = reproducibility(root)
    return [("Frozen protocol", headers, rows)]


def clean_ablation(root: Path) -> list[Block]:
    public_rows = read(root / "public/ablation_summary.json")
    public = []
    public_details = {
        "Full CREDIT": "Semantic fallback deployed",
        "No drift gate": "No version transition in the pinned public suite",
        "No fresh certification": "Certification removed by ablation",
    }
    for row in public_rows:
        lo, hi = row["delta_95ci"]
        detail = public_details.get(row["variant"], "")
        if "raw_uncertified_native_score" in row:
            detail = f"Raw uncertified native={row['raw_uncertified_native_score']:.6f}"
        public.append([
            row["variant"], row["native_score"], row["delta_vs_full"], lo, hi,
            row["total_calls"], row["certification_status"], detail,
        ])

    attribution_diag = read(root / "public/mechanism_diagnostics.json")
    support = read(root / "mechanism_support/summary.json")
    diagnostics = []
    lo, hi = attribution_diag["paired_95ci"]
    diagnostics.append([
        "Ambiguous Memory Attribution", "Randomized exposure", attribution_diag["endpoint"],
        attribution_diag["factorial_aggregate_rmse"], attribution_diag["uniform_aggregate_rmse"],
        attribution_diag["paired_mean_effect"], lo, hi,
        f"{attribution_diag['families']} families; budget={attribution_diag['budget']} calls/family",
    ])
    role = support["role_transport"]
    lo, hi = role["paired_95ci"]
    diagnostics.append([
        "Uncertain Memory Transfer", "Role transport", "Held-out utility prediction MAE",
        role["structured_mae"], role["pooled_mae"], role["pooled_minus_structured_mae"],
        lo, hi,
        f"{role['total_compositions']:,} compositions; Holm p={role['holm_p']:.3e}",
    ])
    drift = support["drift_gate"]
    diagnostics.append([
        "Uncertain Memory Transfer", "Drift gate",
        "Unsafe accepted transfers / divergent opportunities", 0.0, 1.0,
        drift["unsafe_rate_increase_without_gate"], drift["unsafe_rate_paired_95ci"][0],
        drift["unsafe_rate_paired_95ci"][1],
        (f"Full {drift['full_unsafe_accepted']}/{drift['divergent_opportunities']}; "
         f"ablated {drift['no_gate_unsafe_accepted']}/{drift['divergent_opportunities']}; "
         f"full CP95 upper={drift['full_zero_count_cp95'][1]:.4f}; "
         f"stable false blocks={drift['stable_false_blocks']}/{drift['stable_opportunities']}"),
    ])
    cert = support["fresh_certification"]
    full_rate = cert["full_unsafe_accepted"] / cert["full_accepted"]
    ablated_rate = cert["no_certification_unsafe_accepted"] / cert["no_certification_accepted"]
    diagnostics.append([
        "Budget-Constrained Update Reliability", "Fresh certification",
        "Unsafe accepted updates / accepted updates", full_rate, ablated_rate,
        cert["unsafe_rate_increase_without_certification"], cert["paired_95ci"][0],
        cert["paired_95ci"][1],
        (f"Full {cert['full_unsafe_accepted']}/{cert['full_accepted']}; "
         f"ablated {cert['no_certification_unsafe_accepted']}/{cert['no_certification_accepted']}; "
         f"full CP95 upper={cert['full_zero_count_cp95'][1]:.4f}"),
    ])
    return [
        (
            "A. Public benchmark ablation",
            ["Variant", "Native Score", "Delta vs Full", "CI Lower", "CI Upper",
             "Total Calls", "Certification Status", "Deployment Detail"],
            public,
        ),
        (
            "B. Linked controlled diagnostics",
            ["Challenge", "Removed Mechanism", "Endpoint", "Full Metric", "Ablated Metric",
             "Ablated minus Full", "Effect CI Lower", "Effect CI Upper", "Evidence / Sample"],
            diagnostics,
        ),
    ]


CLEAN_BUILDERS = {
    "T1_Public_Main": clean_public_main,
    "T2_Attribution": clean_attribution,
    "T3_Replay": clean_replay,
    "T4_Certification": clean_certification,
    "T5_Budget": clean_budget,
    "T6_Amortization": clean_amortization,
    "T7_Reproducibility": clean_reproducibility,
    "T8_Ablation": clean_ablation,
}


RATE_HEADERS = {
    "VSR Rate", "Coverage Rate", "False Improvement Rate", "Guard Error Rate",
    "Any-cycle Error Rate", "Error CI Lower", "Error CI Upper", "Power 2 pp",
    "2 pp CI Lower", "2 pp CI Upper", "Power 5 pp", "5 pp CI Lower",
    "5 pp CI Upper", "Power 10 pp", "10 pp CI Lower", "10 pp CI Upper",
}


def write_clean_sheet(ws, blocks: list[Block], sheet_index: int) -> None:
    max_col = max(len(headers) for _, headers, _ in blocks)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    ws["A1"] = CAPTIONS[ws.title]
    ws["A1"].font = Font(size=14, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 24
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max_col)
    ws["A2"] = "Source: results/credit_experiment_tables_v1 audited JSON artifacts"
    ws["A2"].font = Font(size=9, italic=True, color="666666")
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=max_col)
    ws["A3"] = NOTES[ws.title]
    ws["A3"].font = Font(size=9, color="444444")
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[3].height = 30

    widths: dict[int, int] = {}
    row_cursor = 5
    thin = Side(style="thin", color="D9E2F3")
    for block_index, (block_title, headers, rows) in enumerate(blocks, 1):
        block_cols = len(headers)
        ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=block_cols)
        title_cell = ws.cell(row_cursor, 1, block_title)
        title_cell.font = Font(bold=True, color="1F1F1F")
        title_cell.fill = PatternFill("solid", fgColor="D9EAF7")
        header_row = row_cursor + 1
        for col, header in enumerate(headers, 1):
            cell = ws.cell(header_row, col, header)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="5B9BD5")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            widths[col] = max(widths.get(col, 10), len(str(header)) + 2)
        ws.row_dimensions[header_row].height = 32

        for row_index, values in enumerate(rows, header_row + 1):
            if len(values) != block_cols:
                raise ValueError(f"{ws.title}/{block_title}: row width mismatch")
            for col, value in enumerate(values, 1):
                cell = ws.cell(row_index, col, value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                cell.border = Border(bottom=thin)
                if isinstance(value, float):
                    cell.number_format = "0.0000"
                if value == "N/A":
                    cell.comment = Comment(
                        "Unsupported estimand for this estimator; the experiment ran, "
                        "but this method does not define this target. This is not a missing result.",
                        "CREDIT experiment pipeline",
                    )
                widths[col] = max(widths.get(col, 10), min(42, len(str(value)) + 2))
            if "Full CREDIT" in values:
                for col in range(1, block_cols + 1):
                    ws.cell(row_index, col).fill = PatternFill("solid", fgColor="E8F2F9")
                    ws.cell(row_index, col).font = Font(bold=True)

        end_row = header_row + len(rows)
        table = Table(
            displayName=f"CREDITTable{sheet_index}_{block_index}",
            ref=f"A{header_row}:{get_column_letter(block_cols)}{end_row}",
        )
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        ws.add_table(table)

        for col, header in enumerate(headers, 1):
            for row_index in range(header_row + 1, end_row + 1):
                cell = ws.cell(row_index, col)
                if not isinstance(cell.value, (int, float)):
                    continue
                if header in RATE_HEADERS:
                    cell.number_format = "0.00%"
                elif header in {"Raw p", "Holm p"}:
                    cell.number_format = "0.0000"
        if ws.title == "T8_Ablation" and block_index == 2:
            removed_col = headers.index("Removed Mechanism") + 1
            rate_cols = [headers.index(name) + 1 for name in (
                "Full Metric", "Ablated Metric", "Ablated minus Full",
                "Effect CI Lower", "Effect CI Upper",
            )]
            for row_index in range(header_row + 1, end_row + 1):
                if ws.cell(row_index, removed_col).value in {"Drift gate", "Fresh certification"}:
                    for col in rate_cols:
                        ws.cell(row_index, col).number_format = "0.00%"
        row_cursor = end_row + 2

    for col in range(1, max_col + 1):
        ws.column_dimensions[get_column_letter(col)].width = min(42, max(10, widths.get(col, 10)))
    ws.freeze_panes = "A7"
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def table_records(ws, table_name: str) -> list[dict[str, Any]]:
    table = ws.tables[table_name]
    min_col, min_row, max_col, max_row = range_boundaries(table.ref)
    headers = [ws.cell(min_row, col).value for col in range(min_col, max_col + 1)]
    return [
        {header: ws.cell(row, col).value for header, col in zip(headers, range(min_col, max_col + 1))}
        for row in range(min_row + 1, max_row + 1)
    ]


def verify_clean(path: Path, root: Path) -> dict:
    wb = load_workbook(path, data_only=False, read_only=False)
    expected_table_counts = {
        "T1_Public_Main": 2,
        "T2_Attribution": 1,
        "T3_Replay": 2,
        "T4_Certification": 3,
        "T5_Budget": 1,
        "T6_Amortization": 1,
        "T7_Reproducibility": 1,
        "T8_Ablation": 2,
    }
    table_counts = {ws.title: len(ws.tables) for ws in wb.worksheets}
    formula_cells = [
        f"{ws.title}!{cell.coordinate}"
        for ws in wb.worksheets for row in ws.iter_rows() for cell in row
        if isinstance(cell.value, str) and cell.value.startswith("=")
    ]
    na_cells = [
        f"{ws.title}!{cell.coordinate}"
        for ws in wb.worksheets for row in ws.iter_rows() for cell in row
        if isinstance(cell.value, str) and cell.value.strip().upper() == "N/A"
    ]
    checks = {
        "sheet_order": tuple(wb.sheetnames) == SHEETS,
        "panel_layout": table_counts == expected_table_counts,
        "no_formula_cells": not formula_cells,
        "no_external_links": not bool(getattr(wb, "_external_links", [])),
        "only_unsupported_estimands_are_na": (
            len(na_cells) == 27 and all(cell.startswith("T2_Attribution!") for cell in na_cells)
        ),
    }
    expected_data_rows = {
        "T1_Public_Main": 10, "T2_Attribution": 15, "T3_Replay": 5,
        "T4_Certification": 12, "T5_Budget": 16, "T6_Amortization": 16,
        "T7_Reproducibility": 28, "T8_Ablation": 9,
    }
    data_rows = {}
    for ws in wb.worksheets:
        data_rows[ws.title] = sum(
            range_boundaries(table.ref)[3] - range_boundaries(table.ref)[1]
            for table in ws.tables.values()
        )
    checks["data_row_counts"] = data_rows == expected_data_rows

    main = {row["method"]: row for row in read(root / "public/main_summary.json")}
    t1_records = table_records(wb["T1_Public_Main"], "CREDITTable1_1")
    full = next(row for row in t1_records if row["Method"] == "Full CREDIT")
    checks["full_credit_native_score"] = full["Native Score"] == main["Full CREDIT"]["native_score"]

    mechanism = read(root / "mechanism_support/summary.json")
    t8_records = table_records(wb["T8_Ablation"], "CREDITTable8_2")
    role = next(row for row in t8_records if row["Removed Mechanism"] == "Role transport")
    checks["role_transport_effect"] = math.isclose(
        role["Ablated minus Full"], mechanism["role_transport"]["pooled_minus_structured_mae"],
        rel_tol=1e-14, abs_tol=1e-15,
    )
    if not all(checks.values()):
        raise AssertionError({"checks": checks, "na_cells": na_cells})
    return {
        "passed": True,
        "workbook": str(path),
        "sha256": workbook_sha(path),
        "sheets": list(SHEETS),
        "excel_table_counts": table_counts,
        "data_row_counts": data_rows,
        "n_a_cells": len(na_cells),
        "n_a_definition": "unsupported estimator target only; never a missing experiment or zero",
        "checks": checks,
    }


def verify(path: Path, root: Path) -> dict:
    wb = load_workbook(path, data_only=False, read_only=False)
    checks = {
        "sheet_order": tuple(wb.sheetnames) == SHEETS,
        "one_excel_table_per_sheet": all(len(ws.tables) == 1 for ws in wb.worksheets),
        "no_formula_cells": not any(
            isinstance(cell.value, str) and cell.value.startswith("=")
            for ws in wb.worksheets for row in ws.iter_rows() for cell in row
        ),
        "no_external_links": not bool(getattr(wb, "_external_links", [])),
    }
    expected_rows = {
        "T1_Public_Main": 10,
        "T2_Attribution": 15,
        "T3_Replay": 5,
        "T4_Certification": 20,
        "T5_Budget": 16,
        "T6_Amortization": 16,
        "T7_Reproducibility": 28,
        "T8_Ablation": 9,
    }
    row_counts = {ws.title: ws.max_row - 5 for ws in wb.worksheets}
    checks["row_counts"] = row_counts == expected_rows

    # Cross-check headline cells against the audited JSON values.
    main = {row["method"]: row for row in read(root / "public/main_summary.json")}
    mechanism = read(root / "mechanism_support/summary.json")
    t1 = wb["T1_Public_Main"]
    headers = {cell.value: cell.column for cell in t1[5]}
    full_row = next(row for row in range(6, t1.max_row + 1)
                    if t1.cell(row, headers["Item"]).value == "Full CREDIT")
    checks["full_credit_native_score"] = (
        t1.cell(full_row, headers["Native Score"]).value == main["Full CREDIT"]["native_score"]
    )
    t8 = wb["T8_Ablation"]
    headers8 = {cell.value: cell.column for cell in t8[5]}
    role_row = next(row for row in range(6, t8.max_row + 1)
                    if t8.cell(row, headers8["Item"]).value == "Role transport")
    checks["role_transport_effect"] = math.isclose(
        t8.cell(role_row, headers8["Ablated minus Full"]).value,
        mechanism["role_transport"]["pooled_minus_structured_mae"],
        rel_tol=1e-14,
        abs_tol=1e-15,
    )
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "passed": True,
        "workbook": str(path),
        "sha256": workbook_sha(path),
        "sheets": list(SHEETS),
        "row_counts": row_counts,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/credit_experiment_tables_v1"))
    parser.add_argument("--output", type=Path, default=Path("CREDIT_Experiment_Results.xlsx"))
    args = parser.parse_args()
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "CREDIT Experiment Results"
    wb.properties.subject = "Audited results exported from CREDIT_Experiment_Tables.tex"
    wb.properties.creator = "CREDIT experiment pipeline"
    wb.properties.description = "One LaTeX result table per worksheet; values generated from audited JSON artifacts."
    for index, name in enumerate(SHEETS, 1):
        ws = wb.create_sheet(name)
        blocks = CLEAN_BUILDERS[name](args.root)
        write_clean_sheet(ws, blocks, index)
    wb.active = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.output)
    audit = verify_clean(args.output, args.root)
    audit_path = args.root / "xlsx_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
