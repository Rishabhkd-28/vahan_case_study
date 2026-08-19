-- Aggregation query for the Vahan lead-source cohort case study.
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
