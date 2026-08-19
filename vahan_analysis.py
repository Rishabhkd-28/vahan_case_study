from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = "/Users/rishabhsharma/Downloads/Vahan_Case_Study (1).xlsx"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator / denominator, np.nan)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Raw Data")
    df.columns = [str(c).strip() for c in df.columns]
    df["upload_date"] = pd.to_datetime(df["upload_date"], errors="coerce")

    numeric_cols = [
        "Uploaded Leads",
        "Attempted",
        "Connected",
        "Attempt per Lead",
        "tag_filled",
        "Interested",
        "OB_after_upload",
        "OB_after_first_attempt",
        "FT_after_upload",
        "FT_after_first_attempt",
        "upload_to_first_attempt_P50 (hrs)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("lead_source", dropna=False)
        .agg(
            upload_start=("upload_date", "min"),
            upload_end=("upload_date", "max"),
            uploaded_leads=("Uploaded Leads", "sum"),
            attempted=("Attempted", "sum"),
            connected=("Connected", "sum"),
            tag_filled=("tag_filled", "sum"),
            interested=("Interested", "sum"),
            ob_after_upload=("OB_after_upload", "sum"),
            ob_after_first_attempt=("OB_after_first_attempt", "sum"),
            ft_after_upload=("FT_after_upload", "sum"),
            ft_after_first_attempt=("FT_after_first_attempt", "sum"),
            total_attempts=("Attempt per Lead", "sum"),
            median_upload_to_first_attempt_hrs=("upload_to_first_attempt_P50 (hrs)", "median"),
            rows=("candidate_phone", "size"),
        )
        .reset_index()
    )

    grouped["attempt_rate"] = rate(grouped["attempted"], grouped["uploaded_leads"])
    grouped["connect_rate_per_attempted"] = rate(grouped["connected"], grouped["attempted"])
    grouped["interest_rate_per_connected"] = rate(grouped["interested"], grouped["connected"])
    grouped["ft_rate_per_uploaded"] = rate(grouped["ft_after_upload"], grouped["uploaded_leads"])
    grouped["ft_rate_per_attempted"] = rate(grouped["ft_after_upload"], grouped["attempted"])
    grouped["ft_after_first_attempt_rate"] = rate(
        grouped["ft_after_first_attempt"], grouped["attempted"]
    )
    grouped["attempts_per_uploaded_lead"] = rate(
        grouped["total_attempts"], grouped["uploaded_leads"]
    )

    min_volume = max(100, int(grouped["uploaded_leads"].median() * 0.5))
    grouped["meets_volume_floor"] = grouped["uploaded_leads"] >= min_volume

    # Composite score balances final conversion quality and usable scale.
    # FT rate is primary; log volume keeps tiny high-rate cohorts from dominating.
    volume_component = np.log1p(grouped["uploaded_leads"]) / np.log1p(
        grouped["uploaded_leads"].max()
    )
    grouped["cohort_score"] = grouped["ft_rate_per_uploaded"].fillna(0) * volume_component
    grouped.loc[~grouped["meets_volume_floor"], "cohort_score"] *= 0.5

    return grouped.sort_values(
        ["cohort_score", "ft_rate_per_uploaded", "ft_after_upload", "uploaded_leads"],
        ascending=False,
    )


def build_ml_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    model_df = df.copy()
    model_df["target_ft_after_upload"] = model_df["FT_after_upload"].fillna(0).astype(int)
    model_df["upload_day_of_week"] = model_df["upload_date"].dt.dayofweek
    model_df["upload_day"] = model_df["upload_date"].dt.day

    # Exclude direct or near-direct post-outcome leakage from features.
    excluded = {
        "candidate_phone",
        "FT_after_upload",
        "FT_after_first_attempt",
        "OB_after_upload",
        "OB_after_first_attempt",
        "Interested → FT_after_first_attempt %",
        "Attempted → FT_after_upload %",
        "upload_date",
        "target_ft_after_upload",
    }
    feature_cols = [
        "lead_source",
        "Uploaded Leads",
        "Attempted",
        "Connected",
        "Attempt per Lead",
        "tag_filled",
        "Interested",
        "upload_to_first_attempt_P50 (hrs)",
        "upload_day_of_week",
        "upload_day",
    ]
    feature_cols = [c for c in feature_cols if c in model_df.columns and c not in excluded]

    categorical = ["lead_source"]
    numeric = [c for c in feature_cols if c not in categorical]

    x = model_df[feature_cols]
    x = x.replace([np.inf, -np.inf], np.nan)
    y = model_df["target_ft_after_upload"]
    return x, y, categorical, numeric


def best_validation_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def train_models(df: pd.DataFrame) -> dict:
    x, y, categorical, numeric = build_ml_dataset(df)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=25),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    fitted = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(x_train, y_train)
        val_proba = pipe.predict_proba(x_val)[:, 1]
        threshold, val_f1 = best_validation_threshold(y_val, val_proba)
        proba = pipe.predict_proba(x_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        fitted[name] = {
            "pipeline": pipe,
            "pred": pred,
            "proba": proba,
            "threshold": threshold,
            "validation_f1_at_threshold": val_f1,
            "metrics": {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
            },
            "confusion_matrix": confusion_matrix(y_test, pred),
            "classification_report": classification_report(y_test, pred, zero_division=0),
        }

    # Prefer recall/F1 for a rare-event conversion model, but keep interpretability in report.
    best_name = max(fitted, key=lambda n: (fitted[n]["metrics"]["f1"], fitted[n]["metrics"]["recall"]))
    best = fitted[best_name]

    scored_test = pd.DataFrame({"actual": y_test.to_numpy(), "score": best["proba"]})
    scored_test = scored_test.sort_values("score", ascending=False).reset_index(drop=True)
    top_10_count = max(1, int(len(scored_test) * 0.10))
    top_20_count = max(1, int(len(scored_test) * 0.20))
    base_rate = scored_test["actual"].mean()
    top_10_rate = scored_test.head(top_10_count)["actual"].mean()
    top_20_rate = scored_test.head(top_20_count)["actual"].mean()
    lift = {
        "base_rate": base_rate,
        "top_10_percent_rate": top_10_rate,
        "top_20_percent_rate": top_20_rate,
        "top_10_percent_lift": top_10_rate / base_rate if base_rate else np.nan,
        "top_20_percent_lift": top_20_rate / base_rate if base_rate else np.nan,
        "positives_in_test": int(scored_test["actual"].sum()),
        "positives_in_top_10_percent": int(scored_test.head(top_10_count)["actual"].sum()),
        "positives_in_top_20_percent": int(scored_test.head(top_20_count)["actual"].sum()),
    }

    feature_names = best["pipeline"].named_steps["preprocess"].get_feature_names_out()
    model = best["pipeline"].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
    else:
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(model.coef_[0])}
        ).sort_values("importance", ascending=False)

    return {
        "best_model_name": best_name,
        "all_models": fitted,
        "best": best,
        "feature_importance": importance,
        "lift": lift,
        "y_test_positive_rate": y_test.mean(),
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "test_rows": len(x_test),
    }


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def write_sql(out_dir: Path) -> None:
    sql = """-- Aggregation query for the Vahan lead-source cohort case study.
-- Assumes the raw data is available as table raw_vahan_leads with columns named as in Excel.

SELECT
    lead_source,
    MIN(upload_date) AS upload_start,
    MAX(upload_date) AS upload_end,
    SUM("Uploaded Leads") AS uploaded_leads,
    SUM("Attempted") AS attempted,
    SUM("Connected") AS connected,
    SUM("tag_filled") AS tag_filled,
    SUM("Interested") AS interested,
    SUM("OB_after_upload") AS ob_after_upload,
    SUM("OB_after_first_attempt") AS ob_after_first_attempt,
    SUM("FT_after_upload") AS ft_after_upload,
    SUM("FT_after_first_attempt") AS ft_after_first_attempt,
    SUM("Attempt per Lead") AS total_attempts,
    SUM("Attempted") * 1.0 / NULLIF(SUM("Uploaded Leads"), 0) AS attempt_rate,
    SUM("Connected") * 1.0 / NULLIF(SUM("Attempted"), 0) AS connect_rate_per_attempted,
    SUM("Interested") * 1.0 / NULLIF(SUM("Connected"), 0) AS interest_rate_per_connected,
    SUM("FT_after_upload") * 1.0 / NULLIF(SUM("Uploaded Leads"), 0) AS ft_rate_per_uploaded,
    SUM("FT_after_upload") * 1.0 / NULLIF(SUM("Attempted"), 0) AS ft_rate_per_attempted,
    SUM("FT_after_first_attempt") * 1.0 / NULLIF(SUM("Attempted"), 0) AS ft_after_first_attempt_rate,
    SUM("Attempt per Lead") * 1.0 / NULLIF(SUM("Uploaded Leads"), 0) AS attempts_per_uploaded_lead
FROM raw_vahan_leads
GROUP BY lead_source
ORDER BY ft_rate_per_uploaded DESC, ft_after_upload DESC, uploaded_leads DESC;
"""
    (out_dir / "aggregation_query.sql").write_text(sql)


def write_charts(out_dir: Path, cohort_table: pd.DataFrame, df: pd.DataFrame) -> None:
    top = cohort_table.head(8).copy()
    max_rate = max(float(top["ft_rate_per_uploaded"].max()), 0.0001)
    rows = []
    height = 90 + len(top) * 42
    for i, row in enumerate(top.itertuples(index=False), start=0):
        y = 65 + i * 42
        width = 520 * float(row.ft_rate_per_uploaded) / max_rate
        label = escape(str(row.lead_source)[:48])
        pct = float(row.ft_rate_per_uploaded) * 100
        rows.append(f'<text x="20" y="{y + 15}" font-size="12" fill="#1f2933">{label}</text>')
        rows.append(f'<rect x="360" y="{y}" width="{width:.1f}" height="22" fill="#2f6f73"/>')
        rows.append(f'<text x="{370 + width:.1f}" y="{y + 16}" font-size="12" fill="#1f2933">{pct:.2f}%</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" viewBox="0 0 980 {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="20" y="32" font-size="22" font-weight="700" fill="#111827">Top Lead-Source Cohorts by FT Rate</text>
{''.join(rows)}
</svg>'''
    (out_dir / "top_cohorts_ft_rate.svg").write_text(svg)

    funnel = pd.Series(
        {
            "Uploaded": df["Uploaded Leads"].sum(),
            "Attempted": df["Attempted"].sum(),
            "Connected": df["Connected"].sum(),
            "Interested": df["Interested"].sum(),
            "FT": df["FT_after_upload"].sum(),
        }
    )
    max_value = max(float(funnel.max()), 1.0)
    bars = []
    for i, (label, value) in enumerate(funnel.items()):
        x = 70 + i * 160
        bar_h = 260 * float(value) / max_value
        y = 340 - bar_h
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="80" height="{bar_h:.1f}" fill="#5b5f97"/>')
        bars.append(f'<text x="{x + 40}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" fill="#111827">{int(value):,}</text>')
        bars.append(f'<text x="{x + 40}" y="370" text-anchor="middle" font-size="13" fill="#1f2933">{label}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="850" height="400" viewBox="0 0 850 400">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="20" y="32" font-size="22" font-weight="700" fill="#111827">Lead Funnel Volume</text>
<line x1="40" y1="340" x2="820" y2="340" stroke="#d1d5db"/>
{''.join(bars)}
</svg>'''
    (out_dir / "funnel_summary.svg").write_text(svg)


def write_report(
    out_dir: Path,
    df: pd.DataFrame,
    cohort_table: pd.DataFrame,
    ml: dict,
) -> None:
    top = cohort_table.head(3).copy()
    overall = {
        "rows": len(df),
        "sources": df["lead_source"].nunique(dropna=True),
        "date_min": df["upload_date"].min().date(),
        "date_max": df["upload_date"].max().date(),
        "uploaded": int(df["Uploaded Leads"].sum()),
        "attempted": int(df["Attempted"].sum()),
        "connected": int(df["Connected"].sum()),
        "interested": int(df["Interested"].sum()),
        "ft_upload": int(df["FT_after_upload"].sum()),
        "ft_first": int(df["FT_after_first_attempt"].sum()),
    }

    cm = ml["best"]["confusion_matrix"]
    metrics = ml["best"]["metrics"]
    importance = ml["feature_importance"].head(12)

    report = []
    report.append("# Vahan Product Analytics Internship Case Study")
    report.append("")
    report.append("## Summary")
    report.append(
        f"The raw dataset contains {overall['rows']:,} lead rows across "
        f"{overall['sources']} lead-source cohorts from {overall['date_min']} to {overall['date_max']}. "
        f"Overall FT-after-upload conversion is low at {format_pct(overall['ft_upload'] / overall['uploaded'])} "
        f"({overall['ft_upload']} FTs from {overall['uploaded']:,} uploaded leads)."
    )
    report.append("")
    report.append(
        "For ranking cohorts, I used FT-after-upload rate as the main metric, with a small volume adjustment. "
        "That keeps the ranking focused on final conversion quality while avoiding tiny cohorts that look good only because of small sample size."
    )
    report.append("")
    report.append("## Top 3 Cohorts")
    report.append("")
    report.append("![Top cohorts by FT rate](top_cohorts_ft_rate.svg)")
    report.append("")
    report.append(
        "| Rank | Lead source | Uploaded leads | FT after upload | FT rate | Attempt rate | Connect rate | Score |"
    )
    report.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(top.itertuples(index=False), start=1):
        report.append(
            f"| {i} | {row.lead_source} | {int(row.uploaded_leads):,} | "
            f"{int(row.ft_after_upload):,} | {format_pct(row.ft_rate_per_uploaded)} | "
            f"{format_pct(row.attempt_rate)} | {format_pct(row.connect_rate_per_attempted)} | "
            f"{row.cohort_score:.4f} |"
        )
    report.append("")
    report.append("## Funnel Readout")
    report.append("")
    report.append("![Lead funnel volume](funnel_summary.svg)")
    report.append("")
    report.append(f"- Uploaded leads: {overall['uploaded']:,}")
    report.append(f"- Attempted: {overall['attempted']:,} ({format_pct(overall['attempted'] / overall['uploaded'])})")
    report.append(f"- Connected: {overall['connected']:,} ({format_pct(overall['connected'] / overall['attempted'])} of attempted)")
    report.append(f"- Interested: {overall['interested']:,} ({format_pct(overall['interested'] / overall['connected'])} of connected)")
    report.append(f"- FT after upload: {overall['ft_upload']:,} ({format_pct(overall['ft_upload'] / overall['uploaded'])} of uploaded)")
    report.append(f"- FT after first attempt: {overall['ft_first']:,} ({format_pct(overall['ft_first'] / overall['attempted'])} of attempted)")
    report.append("")
    report.append("## SQL / Aggregate Table")
    report.append("")
    report.append(
        "The SQL query is saved in `aggregation_query.sql`. The aggregate output is saved in "
        "`cohort_aggregation.csv` and `vahan_case_study_outputs.xlsx`."
    )
    report.append("")
    report.append("## FT Prediction Model")
    report.append("")
    report.append(
        f"I treated `FT_after_upload` as the target. The better performing model was `{ml['best_model_name']}`, "
        f"with the decision threshold tuned on a validation split at {ml['best']['threshold']:.2f}. "
        "Since only a small fraction of leads become FT, I would not judge this model by accuracy alone. "
        "The more useful readout is whether the score can help prioritize a smaller set of leads."
    )
    report.append("")
    report.append("| Metric | Value |")
    report.append("|---|---:|")
    for key, value in metrics.items():
        report.append(f"| {key} | {value:.4f} |")
    report.append("")
    report.append("Confusion matrix on 20% holdout test set:")
    report.append("")
    report.append("| Actual \\ Predicted | Not FT | FT |")
    report.append("|---|---:|---:|")
    report.append(f"| Not FT | {cm[0,0]} | {cm[0,1]} |")
    report.append(f"| FT | {cm[1,0]} | {cm[1,1]} |")
    report.append("")
    report.append("Main factors picked up by the model:")
    report.append("")
    report.append("| Rank | Feature | Importance |")
    report.append("|---:|---|---:|")
    for i, row in enumerate(importance.itertuples(index=False), start=1):
        report.append(f"| {i} | {row.feature} | {row.importance:.4f} |")
    report.append("")
    lift = ml["lift"]
    report.append("Prioritization readout:")
    report.append("")
    report.append(
        f"- Test-set base FT rate: {format_pct(lift['base_rate'])} "
        f"({lift['positives_in_test']} positives)."
    )
    report.append(
        f"- Top 10% scored leads: {format_pct(lift['top_10_percent_rate'])} FT rate, "
        f"{lift['top_10_percent_lift']:.2f}x lift, "
        f"{lift['positives_in_top_10_percent']} captured positives."
    )
    report.append(
        f"- Top 20% scored leads: {format_pct(lift['top_20_percent_rate'])} FT rate, "
        f"{lift['top_20_percent_lift']:.2f}x lift, "
        f"{lift['positives_in_top_20_percent']} captured positives."
    )
    report.append("")
    report.append("## Assumptions")
    report.append("")
    report.append("- `FT_after_upload` is treated as the main business outcome.")
    report.append("- Cohorts are aggregated at `lead_source` level because the brief asks for lead-source cohort performance.")
    report.append("- Post-outcome fields such as `FT_after_first_attempt`, `OB_after_upload`, and conversion-to-FT percentage columns are excluded from model features to reduce leakage.")
    report.append("- FT is a rare outcome in this sample, so the model is best used for prioritization, not as a final automated decision rule.")
    report.append("- If the real constraint is caller capacity, I would also track attempted volume and connect rate alongside FT rate.")
    report.append("")
    report.append("## Recommendation")
    report.append("")
    report.append(
        "I would prioritize the top 3 cohorts above for near-term sourcing, with weekly monitoring so the ranking does not overreact to small changes. "
        "FT rate should be the quality metric, while attempt and connect rates should be used to diagnose execution issues in the calling funnel."
    )
    report.append("")

    (out_dir / "vahan_case_study_report.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default="vahan_case_study/output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    cohort_table = build_cohort_table(df)
    ml = train_models(df)

    cohort_table.to_csv(out_dir / "cohort_aggregation.csv", index=False)
    ml["feature_importance"].to_csv(out_dir / "model_feature_importance.csv", index=False)

    metrics_rows = []
    for model_name, payload in ml["all_models"].items():
        row = {
            "model": model_name,
            "threshold": payload["threshold"],
            "validation_f1_at_threshold": payload["validation_f1_at_threshold"],
        }
        row.update(payload["metrics"])
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "model_metrics.csv", index=False)

    cm = ml["best"]["confusion_matrix"]
    pd.DataFrame(
        cm,
        index=["Actual_Not_FT", "Actual_FT"],
        columns=["Pred_Not_FT", "Pred_FT"],
    ).to_csv(out_dir / "confusion_matrix.csv")

    write_sql(out_dir)
    write_charts(out_dir, cohort_table, df)
    write_report(out_dir, df, cohort_table, ml)

    with pd.ExcelWriter(out_dir / "vahan_case_study_outputs.xlsx", engine="openpyxl") as writer:
        cohort_table.to_excel(writer, sheet_name="Cohort Aggregation", index=False)
        cohort_table.head(3).to_excel(writer, sheet_name="Top 3 Cohorts", index=False)
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="Model Metrics", index=False)
        pd.DataFrame(
            cm,
            index=["Actual_Not_FT", "Actual_FT"],
            columns=["Pred_Not_FT", "Pred_FT"],
        ).to_excel(writer, sheet_name="Confusion Matrix")
        pd.DataFrame([ml["lift"]]).to_excel(writer, sheet_name="Model Lift", index=False)
        ml["feature_importance"].head(30).to_excel(
            writer, sheet_name="Feature Importance", index=False
        )

    print("Done.")
    print(f"Report: {out_dir / 'vahan_case_study_report.md'}")
    print(f"Workbook: {out_dir / 'vahan_case_study_outputs.xlsx'}")
    print(f"SQL: {out_dir / 'aggregation_query.sql'}")


if __name__ == "__main__":
    main()
