-- BankLens dbt Models
-- Transform raw bank panel data into analytical tables

-- stg_bank_panel.sql
{{ config(materialized='view') }}
SELECT
    bank_id,
    instname,
    quarter_date,
    total_assets,
    npa_ratio,
    deposit_growth_qoq,
    loan_to_deposit_ratio,
    net_interest_margin,
    charge_off_rate,
    capital_ratio,
    is_stress_bank,
    npa_change,
    ldr_change,
    nim_change,
    deposit_outflow_flag
FROM {{ source('banklens', 'bank_panel') }}
WHERE quarter_date IS NOT NULL

-- mart_bank_stress_scores.sql
{{ config(materialized='table') }}
WITH base AS (
    SELECT * FROM {{ ref('stg_bank_panel') }}
),
stress_signals AS (
    SELECT
        bank_id,
        instname,
        quarter_date,
        total_assets,
        npa_ratio,
        deposit_growth_qoq,
        loan_to_deposit_ratio,
        net_interest_margin,
        charge_off_rate,
        capital_ratio,
        npa_change,
        ldr_change,
        nim_change,
        deposit_outflow_flag,
        -- Composite stress indicators
        CASE
            WHEN npa_ratio > 0.05 THEN 'High NPA'
            WHEN npa_ratio > 0.02 THEN 'Elevated NPA'
            ELSE 'Normal NPA'
        END AS npa_category,
        CASE
            WHEN deposit_growth_qoq < -0.05 THEN 'Significant Outflow'
            WHEN deposit_growth_qoq < -0.02 THEN 'Moderate Outflow'
            WHEN deposit_growth_qoq < 0 THEN 'Slight Outflow'
            ELSE 'Stable/Growing'
        END AS deposit_category,
        CASE
            WHEN loan_to_deposit_ratio > 1.0 THEN 'Over-leveraged'
            WHEN loan_to_deposit_ratio > 0.85 THEN 'Elevated LDR'
            ELSE 'Normal LDR'
        END AS ldr_category
    FROM base
)
SELECT
    *,
    -- Early warning flag: 2+ stress signals simultaneously
    CASE
        WHEN npa_category != 'Normal NPA'
             AND deposit_category IN ('Significant Outflow', 'Moderate Outflow')
        THEN 1 ELSE 0
    END AS early_warning_flag,
    -- Quarter-over-quarter stress escalation
    CASE
        WHEN npa_change > 0.01 AND ldr_change > 0.05 THEN 'Escalating'
        WHEN npa_change > 0.005 THEN 'Deteriorating'
        WHEN npa_change < -0.005 THEN 'Improving'
        ELSE 'Stable'
    END AS stress_trajectory
FROM stress_signals

-- mart_executive_summary.sql
{{ config(materialized='table') }}
SELECT
    quarter_date,
    COUNT(DISTINCT bank_id) AS n_banks,
    AVG(npa_ratio) AS avg_npa_ratio,
    AVG(deposit_growth_qoq) AS avg_deposit_growth,
    AVG(loan_to_deposit_ratio) AS avg_ldr,
    AVG(net_interest_margin) AS avg_nim,
    SUM(deposit_outflow_flag) AS banks_with_outflow,
    SUM(CASE WHEN npa_ratio > 0.05 THEN 1 ELSE 0 END) AS high_npa_count,
    AVG(capital_ratio) AS avg_capital_ratio
FROM {{ ref('stg_bank_panel') }}
GROUP BY quarter_date
ORDER BY quarter_date
