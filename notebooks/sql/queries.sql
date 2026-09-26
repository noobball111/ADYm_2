
-- Step 3: sql/queries.sql
CREATE OR REPLACE TABLE sol AS SELECT * FROM 'data/processed/solar.parquet';
-- Q1: capacity factor by hour and month per plant
SELECT plant, hour(ts) AS hr, month(ts) AS mon, AVG(cf) AS cf FROM sol GROUP BY 1, 2, 3;
-- Q2: variability by cloud group
SELECT CASE WHEN cloud < 20 THEN 'clear' WHEN cloud < 70 THEN 'partly' ELSE 'overcast' END AS sky,
       corr(power, GHI) AS corr_ghi, STDDEV(cf) AS sd_cf, COUNT(*) AS n
FROM sol GROUP BY 1;
-- Q3: features and targets 1 and 6 hours ahead
CREATE OR REPLACE TABLE feat AS
SELECT plant, ts, cf, GHI, DNI, cloud, temp, hour(ts) AS hr, month(ts) AS mon,
       LAG(cf, 1) OVER w AS cf_lag1, LAG(cf, 24) OVER w AS cf_lag24,
       AVG(cf) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS cf_ma3,
       LEAD(cf, 1) OVER w AS y_1h, LEAD(cf, 6) OVER w AS y_6h
FROM sol WINDOW w AS (PARTITION BY plant ORDER BY ts);
SELECT COUNT(*) AS n, COUNT(y_6h) AS n6 FROM feat;
