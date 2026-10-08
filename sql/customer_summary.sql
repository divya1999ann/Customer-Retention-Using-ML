-- Supporting SQL for the causal customer-retention project.
-- Run from the notebooks against an in-memory SQLite database built from the dataset:
--   customers(customer_id, FACTOR*, PC1..PC3)  -- customer attributes (covariates X)
--   campaign(customer_id, t, y)                -- experiment log (treatment T, outcome Y)
--   scores(customer_id, uplift_score, churn_risk)  -- model outputs (notebook 06 only)
-- Each query is preceded by a "-- name: <query_name>" line.

-- name: arm_summary
-- Treatment/control sizes and churn rates (the basic experimental readout).
SELECT
    CASE WHEN t = 1 THEN 'treated' ELSE 'control' END AS arm,
    COUNT(*)                                          AS customers,
    SUM(y)                                            AS churners,
    ROUND(AVG(y), 4)                                  AS churn_rate
FROM campaign
GROUP BY t
ORDER BY t;

-- name: segment_assignment
-- Join attributes to the experiment log and summarise each segment of a factor:
-- share treated, churn by arm, and the raw (unadjusted) churn difference.
-- Segments with fewer than 100 customers are filtered out with HAVING.
SELECT
    c.FACTOR18                                                   AS segment,
    COUNT(*)                                                     AS customers,
    ROUND(AVG(e.t), 3)                                           AS treated_share,
    ROUND(AVG(CASE WHEN e.t = 0 THEN e.y END), 4)                AS churn_control,
    ROUND(AVG(CASE WHEN e.t = 1 THEN e.y END), 4)                AS churn_treated,
    ROUND(AVG(CASE WHEN e.t = 1 THEN e.y END)
        - AVG(CASE WHEN e.t = 0 THEN e.y END), 4)                AS raw_difference
FROM customers AS c
JOIN campaign  AS e USING (customer_id)
GROUP BY c.FACTOR18
HAVING COUNT(*) >= 100
ORDER BY treated_share;

-- name: usage_band_churn
-- Customer segmentation on the first principal component (the dominant axis of the
-- numeric usage/billing variables): churn by arm within quartile bands.
WITH banded AS (
    SELECT c.customer_id,
           NTILE(4) OVER (ORDER BY c.PC1) AS pc1_quartile
    FROM customers AS c
)
SELECT
    b.pc1_quartile,
    COUNT(*)                                        AS customers,
    ROUND(AVG(CASE WHEN e.t = 0 THEN e.y END), 4)   AS churn_control,
    ROUND(AVG(CASE WHEN e.t = 1 THEN e.y END), 4)   AS churn_treated
FROM banded   AS b
JOIN campaign AS e USING (customer_id)
GROUP BY b.pc1_quartile
ORDER BY b.pc1_quartile;

-- name: uplift_deciles
-- Notebook 06: rank customers by model score and compare churn between arms within
-- each decile (decile 1 = highest predicted churn reduction).
WITH ranked AS (
    SELECT s.customer_id,
           s.uplift_score,
           NTILE(10) OVER (ORDER BY s.uplift_score DESC) AS decile
    FROM scores AS s
)
SELECT
    r.decile,
    COUNT(*)                                          AS customers,
    SUM(e.t)                                          AS treated,
    ROUND(AVG(CASE WHEN e.t = 0 THEN e.y END), 4)     AS churn_control,
    ROUND(AVG(CASE WHEN e.t = 1 THEN e.y END), 4)     AS churn_treated,
    ROUND(AVG(CASE WHEN e.t = 0 THEN e.y END)
        - AVG(CASE WHEN e.t = 1 THEN e.y END), 4)     AS observed_reduction
FROM ranked   AS r
JOIN campaign AS e USING (customer_id)
GROUP BY r.decile
ORDER BY r.decile;
