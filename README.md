# Vahan Lead-Source Cohort Analysis

This repository contains my solution for the **Vahan Product Analytics Internship Case Study**. The goal was to analyze lead-source cohort performance, identify the best cohorts, write an aggregate SQL query, and build a model to understand what affects the chance of a lead converting to FT.

## Case Study Questions

1. Find the top 3 cohorts and explain the metric used.
2. Write a query to aggregate the raw data at the right level and provide the aggregate output.
3. Build an ML model showing factors that influence the chances of FT, including a confusion matrix.

## Key Findings

I used **FT-after-upload rate with a volume adjustment** as the main ranking metric. FT rate directly answers the business question, while the volume adjustment avoids over-ranking very small cohorts.

Top 3 cohorts:

| Rank | Lead Source | Uploaded Leads | FT After Upload | FT Rate |
|---:|---|---:|---:|---:|
| 1 | Single Referral > 7 days- 24th Jul | 1,500 | 14 | 0.93% |
| 2 | Khanna- 2W 26th Jul | 1,546 | 14 | 0.91% |
| 3 | PreOb-Ob Fees Paid 29th Jul (set 1) | 1,483 | 7 | 0.47% |

Overall funnel:

| Stage | Count | Rate |
|---|---:|---:|
| Uploaded Leads | 18,198 | 100.00% |
| Attempted | 11,973 | 65.79% of uploaded |
| Connected | 5,550 | 46.35% of attempted |
| Interested | 348 | 6.27% of connected |
| FT After Upload | 54 | 0.30% of uploaded |
| FT After First Attempt | 17 | 0.14% of attempted |

## Model Summary

The model target is `FT_after_upload`.

I trained Logistic Regression and Random Forest models, then selected the better model based on F1/recall because FT is a rare outcome. The final selected model was:

**Logistic Regression**

| Metric | Value |
|---|---:|
| Accuracy | 0.9423 |
| Precision | 0.0050 |
| Recall | 0.0909 |
| F1 | 0.0094 |
| ROC-AUC | 0.7603 |

Confusion matrix on the holdout test set:

| Actual \ Predicted | Not FT | FT |
|---|---:|---:|
| Not FT | 3429 | 200 |
| FT | 10 | 1 |

Because only 54 out of 18,198 leads converted to FT, the model is better interpreted as a **lead-prioritization model** rather than a production-ready classifier. The top 10% of scored leads showed about **2.73x lift** over the base FT rate.

## Repository Structure

```text
vahan_case_study/
├── README.md
├── requirements.txt
├── .gitignore
├── vahan_analysis.py
└── output/
    ├── vahan_case_study_report.md
    ├── vahan_case_study_outputs.xlsx
    ├── aggregation_query.sql
    ├── cohort_aggregation.csv
    ├── model_metrics.csv
    ├── confusion_matrix.csv
    ├── model_feature_importance.csv
    ├── top_cohorts_ft_rate.svg
    └── funnel_summary.svg
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python vahan_analysis.py --input "/path/to/Vahan_Case_Study.xlsx"
```

By default, the script writes all outputs to:

```text
vahan_case_study/output/
```

## Important Assumptions

- `FT_after_upload` is used as the main conversion target.
- Cohorts are aggregated at `lead_source` level.
- Fields that directly reveal or closely leak FT outcomes are excluded from the model.
- Since FT conversion is rare, model results should be used directionally for prioritization.

## Final Recommendation

Prioritize the top 3 cohorts for near-term sourcing, but monitor the ranking weekly. FT rate should be the main quality metric, while attempt and connect rates should be tracked to catch execution issues in the calling funnel.
