-- ============================================================
-- MySQL 8+ Date and Time Functions Interview Practice (Varied)
-- Schema: practice
-- STR_TO_DATE('2024-01-31 12:35:23','%Y-%m-%d %H:%i:%s')
-- DATE_ADD
-- DATE_SUB
-- LAST_DAY
-- EXTRACT (YEAR FROM order_date) /MONTH/DAY/HOUR...
-- TIMESTAMPDIFF(MONTH, signup_date, CURDATE()) AS tenure_months

-- ============================================================

USE practice;

-- Q1: Extract multiple components from order_date.
SELECT
  order_id,
  order_date,
  EXTRACT(YEAR FROM order_date) AS yr,
  EXTRACT(MONTH FROM order_date) AS mon,
  EXTRACT(DAY FROM order_date) AS day_no,
  EXTRACT(HOUR FROM order_date) AS hr
FROM orders;

-- Q2: Month start and month end from order_date.
SELECT
  order_id,
  order_date,
  DATE_FORMAT(order_date, '%Y-%m-01') AS month_start,
  LAST_DAY(order_date) AS month_end
FROM orders;

-- Q3: Compute days in month for each order.
SELECT
  order_id,
  order_date,
  DAY(LAST_DAY(order_date)) AS days_in_month
FROM orders;

-- Q4: Add/subtract intervals from signup_date.
SELECT
  customer_id,
  signup_date,
  DATE_ADD(signup_date, INTERVAL 30 DAY) AS plus_30d,
  DATE_SUB(signup_date, INTERVAL 2 MONTH) AS minus_2m
FROM customers;

-- Q5: Business age of customers in months.
SELECT
  customer_id,
  signup_date,
  TIMESTAMPDIFF(MONTH, signup_date, CURDATE()) AS tenure_months
FROM customers;

-- Q6: Payment latency in minutes from order placement.
SELECT
  p.payment_id,
  p.order_id,
  TIMESTAMPDIFF(MINUTE, o.order_date, p.payment_date) AS payment_latency_min
FROM payments p
JOIN orders o ON o.order_id = p.order_id;

-- Q7: Week-based grouping (ISO-like week mode).
SELECT
  YEAR(order_date) AS yr,
  WEEK(order_date, 3) AS iso_week,
  COUNT(*) AS orders_cnt,
  SUM(total_amount) AS sales
FROM orders
GROUP BY YEAR(order_date), WEEK(order_date, 3)
ORDER BY yr, iso_week;

-- Q8: Weekend vs weekday split using DAYOFWEEK.
SELECT
  CASE WHEN DAYOFWEEK(order_date) IN (1,7) THEN 'weekend' ELSE 'weekday' END AS day_type,
  COUNT(*) AS orders_cnt,
  ROUND(AVG(total_amount), 2) AS avg_amt
FROM orders
GROUP BY day_type;

-- Q9: Create hour bucket from order timestamp.
SELECT
  order_id,
  order_date,
  CONCAT(LPAD(HOUR(order_date), 2, '0'), ':00-', LPAD(HOUR(order_date), 2, '0'), ':59') AS hour_bucket
FROM orders;

-- Q10: Month-over-month sales change with lag.
WITH m AS (
  SELECT DATE_FORMAT(order_date, '%Y-%m-01') AS month_start, SUM(total_amount) AS sales
  FROM orders
  GROUP BY DATE_FORMAT(order_date, '%Y-%m-01')
)
SELECT
  month_start,
  sales,
  LAG(sales) OVER (ORDER BY month_start) AS prev_sales,
  sales - LAG(sales) OVER (ORDER BY month_start) AS mom_diff
FROM m
ORDER BY month_start;

-- Q11: Parse date string with STR_TO_DATE.
SELECT STR_TO_DATE('2026-03-25 18:45:00', '%Y-%m-%d %H:%i:%s') AS parsed_dt;

-- Q12: Format date/time for reporting output.
SELECT
  order_id,
  DATE_FORMAT(order_date, '%d-%b-%Y %h:%i %p') AS formatted_order_dt
FROM orders;

-- Q13: Ticket aging in buckets by hours since creation.
SELECT
  ticket_id,
  created_at,
  CASE
    WHEN TIMESTAMPDIFF(HOUR, created_at, NOW()) < 24 THEN 'lt_24h'
    WHEN TIMESTAMPDIFF(HOUR, created_at, NOW()) < 72 THEN '24_to_72h'
    ELSE 'gt_72h'
  END AS age_bucket
FROM support_tickets;

-- Q14: Find customers whose last order month differs from current month.
WITH last_o AS (
  SELECT customer_id, MAX(order_date) AS last_order_date
  FROM orders
  GROUP BY customer_id
)
SELECT customer_id, last_order_date
FROM last_o
WHERE DATE_FORMAT(last_order_date, '%Y-%m') <> DATE_FORMAT(CURDATE(), '%Y-%m');

-- Q15: Difference in years between earliest and latest hire date.
SELECT
  MIN(hire_date) AS min_hire,
  MAX(hire_date) AS max_hire,
  TIMESTAMPDIFF(YEAR, MIN(hire_date), MAX(hire_date)) AS hire_span_years
FROM employees;
