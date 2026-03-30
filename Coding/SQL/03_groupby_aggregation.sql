-- ============================================================
-- MySQL 8+ GROUP BY and Aggregation Interview Practice (Varied)
-- Schema: practice
-- ============================================================

USE practice;

-- Q1: Basic grouping - orders and amount by status.
SELECT order_status, COUNT(*) AS orders_cnt, SUM(total_amount) AS amount_sum
FROM orders
GROUP BY order_status;

-- Q2: Multi-column grouping - monthly + channel sales.
SELECT DATE_FORMAT(order_date, '%Y-%m-01') AS month_start, sales_channel, SUM(total_amount) AS sales
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m-01'), sales_channel
ORDER BY month_start, sales_channel;

-- Q3: DISTINCT aggregate - active customers per city.
SELECT city, COUNT(DISTINCT customer_id) AS active_customers
FROM customers
WHERE is_active = 1
GROUP BY city
ORDER BY active_customers DESC;

-- Q4: HAVING with ratio - payment success rate by method (minimum 100 rows).
SELECT
  payment_method,
  COUNT(*) AS total_txn,
  ROUND(100 * SUM(payment_status = 'success') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM payments
GROUP BY payment_method
HAVING COUNT(*) >= 100;

-- Q5: Aggregate expression - line revenue by category.
SELECT
  c.category_name,
  ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)), 2) AS category_revenue
FROM categories c
JOIN products p ON p.category_id = c.category_id
JOIN order_items oi ON oi.product_id = p.product_id
GROUP BY c.category_id, c.category_name
ORDER BY category_revenue DESC;

-- Q6: Conditional aggregation pivot - ticket counts by status per issue type.
SELECT
  issue_type,
  SUM(ticket_status = 'open') AS open_cnt,
  SUM(ticket_status = 'in_progress') AS in_progress_cnt,
  SUM(ticket_status = 'resolved') AS resolved_cnt,
  SUM(ticket_status = 'closed') AS closed_cnt
FROM support_tickets
GROUP BY issue_type;

-- Q7: GROUP BY with LEFT JOIN - all departments with employee stats.
SELECT
  d.department_name,
  COUNT(e.employee_id) AS emp_cnt,
  ROUND(AVG(e.base_salary), 2) AS avg_salary,
  MIN(e.base_salary) AS min_salary,
  MAX(e.base_salary) AS max_salary
FROM departments d
LEFT JOIN employees e ON e.department_id = d.department_id
GROUP BY d.department_id, d.department_name;

-- Q8: Bucketed aggregation - orders by value band.
SELECT
  CASE
    WHEN total_amount < 1000 THEN 'lt_1k'
    WHEN total_amount < 3000 THEN '1k_to_3k'
    WHEN total_amount < 6000 THEN '3k_to_6k'
    ELSE '6k_plus'
  END AS amount_band,
  COUNT(*) AS order_cnt,
  ROUND(AVG(total_amount), 2) AS avg_amount
FROM orders
GROUP BY amount_band
ORDER BY avg_amount;

-- Q9: Grouping sets style via ROLLUP - sales by year/month plus totals.
SELECT
  YEAR(order_date) AS order_year,
  MONTH(order_date) AS order_month,
  SUM(total_amount) AS sales
FROM orders
GROUP BY YEAR(order_date), MONTH(order_date) WITH ROLLUP;

-- Q10: Top-N per group using aggregate + window (customer spend ranking within segment).
WITH cust_spend AS (
  SELECT c.customer_segment, c.customer_id, SUM(o.total_amount) AS spend
  FROM customers c
  JOIN orders o ON o.customer_id = c.customer_id
  GROUP BY c.customer_segment, c.customer_id
)
SELECT *
FROM (
  SELECT
    cs.*,
    ROW_NUMBER() OVER (PARTITION BY cs.customer_segment ORDER BY cs.spend DESC, cs.customer_id) AS rn
  FROM cust_spend cs
) x
WHERE rn <= 3
ORDER BY customer_segment, rn;

-- Q11: Aggregation + correlated HAVING - cities above overall city average order value.
WITH city_stats AS (
  SELECT c.city, AVG(o.total_amount) AS city_avg
  FROM customers c
  JOIN orders o ON o.customer_id = c.customer_id
  GROUP BY c.city
)
SELECT city, city_avg
FROM city_stats
WHERE city_avg > (SELECT AVG(city_avg) FROM city_stats)
ORDER BY city_avg DESC;

-- Q12: Approx median per channel using ordered rows and COUNT window.
WITH ordered AS (
  SELECT
    sales_channel,
    total_amount,
    ROW_NUMBER() OVER (PARTITION BY sales_channel ORDER BY total_amount) AS rn,
    COUNT(*) OVER (PARTITION BY sales_channel) AS cnt
  FROM orders
)
SELECT
  sales_channel,
  AVG(total_amount) AS median_amount
FROM ordered
WHERE rn IN (FLOOR((cnt + 1) / 2), FLOOR((cnt + 2) / 2))
GROUP BY sales_channel;

-- Q13: Weighted average discount by product.
SELECT
  product_id,
  ROUND(SUM(discount_pct * quantity) / NULLIF(SUM(quantity), 0), 4) AS weighted_discount_pct
FROM order_items
GROUP BY product_id
ORDER BY weighted_discount_pct DESC;

-- Q14: Aggregation across joined dimensions - monthly sales by employee department.
SELECT
  DATE_FORMAT(o.order_date, '%Y-%m-01') AS month_start,
  d.department_name,
  SUM(o.total_amount) AS sales
FROM orders o
JOIN employees e ON e.employee_id = o.employee_id
JOIN departments d ON d.department_id = e.department_id
GROUP BY DATE_FORMAT(o.order_date, '%Y-%m-01'), d.department_name
ORDER BY month_start, d.department_name;

-- Q15: Group-level filtering by standard deviation - channels with high variability.
SELECT
  sales_channel,
  ROUND(STDDEV_POP(total_amount), 2) AS std_amt
FROM orders
GROUP BY sales_channel
HAVING STDDEV_POP(total_amount) > 1000
ORDER BY std_amt DESC;
