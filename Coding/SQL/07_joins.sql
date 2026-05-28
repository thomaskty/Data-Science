-- ============================================================
-- MySQL 8+ Join Interview Practice (Varied Patterns)
-- Schema: practice
-- ============================================================

USE practice;

-- Q1: INNER JOIN - orders with customer identity and segment.
SELECT o.order_id, o.order_date, c.customer_id, c.full_name, c.customer_segment, o.total_amount
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id;

-- Q2: JOIN ... USING - order_items and products on common key.
SELECT oi.order_item_id, oi.order_id, oi.product_id, p.product_name, p.unit_price
FROM order_items oi
JOIN products p USING (product_id);

-- Q3: LEFT JOIN + NULL detection (anti join) - customers who never ordered.
SELECT c.customer_id, c.full_name
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE o.order_id IS NULL;

-- Q4: LEFT JOIN + aggregation - products with sold quantity, including unsold.
SELECT p.product_id, p.product_name, COALESCE(SUM(oi.quantity), 0) AS qty_sold
FROM products p
LEFT JOIN order_items oi ON oi.product_id = p.product_id
GROUP BY p.product_id, p.product_name
ORDER BY qty_sold DESC;

-- Q5: SELF JOIN hierarchy - employee, manager, and skip-level manager.
SELECT
  e.employee_id,
  CONCAT(e.first_name, ' ', e.last_name) AS employee_name,
  m.employee_id AS manager_id,
  CONCAT(m.first_name, ' ', m.last_name) AS manager_name,
  sm.employee_id AS skip_manager_id,
  CONCAT(sm.first_name, ' ', sm.last_name) AS skip_manager_name
FROM employees e
LEFT JOIN employees m ON m.employee_id = e.manager_id
LEFT JOIN employees sm ON sm.employee_id = m.manager_id;

-- Q6: Multi-table join - order -> item -> product -> category.
SELECT
  o.order_id,
  o.order_date,
  oi.order_item_id,
  p.product_name,
  c.category_name,
  ROUND(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100), 2) AS line_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN categories c ON c.category_id = p.category_id;

-- Q7: Non-equi JOIN - place each order in a manually defined order-value band.
SELECT
  o.order_id,
  o.total_amount,
  b.band_name
FROM orders o
JOIN (
  SELECT 'low' AS band_name, 0 AS min_amt, 999.99 AS max_amt
  UNION ALL SELECT 'mid', 1000, 2999.99
  UNION ALL SELECT 'high', 3000, 5999.99
  UNION ALL SELECT 'vip', 6000, 99999999
) b
  ON o.total_amount BETWEEN b.min_amt AND b.max_amt;

-- Q8: Join with additional ON-condition - orders placed after customer signup by at least 30 days.
SELECT o.order_id, o.customer_id, c.signup_date, o.order_date
FROM orders o
JOIN customers c
  ON c.customer_id = o.customer_id
 AND o.order_date >= DATE_ADD(c.signup_date, INTERVAL 30 DAY);

-- Q9: Semi-join via EXISTS - customers with at least one successful payment.
SELECT c.customer_id, c.full_name
FROM customers c
WHERE EXISTS (
  SELECT 1
  FROM orders o
  JOIN payments p ON p.order_id = o.order_id
  WHERE o.customer_id = c.customer_id
    AND p.payment_status = 'success'
);

-- Q10: NOT EXISTS - employees who never handled cancelled/returned orders.
SELECT e.employee_id, CONCAT(e.first_name, ' ', e.last_name) AS employee_name
FROM employees e
WHERE NOT EXISTS (
  SELECT 1
  FROM orders o
  WHERE o.employee_id = e.employee_id
    AND o.order_status IN ('cancelled', 'returned')
);

-- Q11: CROSS JOIN template - all department x channel combinations with observed order count.
SELECT
  d.department_name,
  ch.sales_channel,
  COUNT(o.order_id) AS orders_cnt
FROM departments d
CROSS JOIN (
  SELECT 'web' AS sales_channel
  UNION ALL SELECT 'mobile_app'
  UNION ALL SELECT 'store'
  UNION ALL SELECT 'partner'
) ch
LEFT JOIN employees e ON e.department_id = d.department_id
LEFT JOIN orders o
  ON o.employee_id = e.employee_id
 AND o.sales_channel = ch.sales_channel
GROUP BY d.department_name, ch.sales_channel
ORDER BY d.department_name, ch.sales_channel;

-- Q12: FULL OUTER JOIN emulation (UNION ALL) - customers and tickets, including unmatched rows.
SELECT c.customer_id, c.full_name, t.ticket_id
FROM customers c
LEFT JOIN support_tickets t ON t.customer_id = c.customer_id
UNION ALL
SELECT c.customer_id, c.full_name, t.ticket_id
FROM support_tickets t
LEFT JOIN customers c ON c.customer_id = t.customer_id
WHERE c.customer_id IS NULL;

-- Q13: Join + window - latest successful payment per customer.
WITH pay_rank AS (
  SELECT
    o.customer_id,
    p.payment_id,
    p.payment_date,
    p.amount,
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY p.payment_date DESC, p.payment_id DESC) AS rn
  FROM payments p
  JOIN orders o ON o.order_id = p.order_id
  WHERE p.payment_status = 'success'
)
SELECT customer_id, payment_id, payment_date, amount
FROM pay_rank
WHERE rn = 1;

-- Q14: Join + HAVING - categories where avg line revenue exceeds overall avg line revenue.
WITH line_rev AS (
  SELECT p.category_id, (oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS rev
  FROM order_items oi
  JOIN products p ON p.product_id = oi.product_id
)
SELECT c.category_name, AVG(lr.rev) AS avg_cat_line_rev
FROM line_rev lr
JOIN categories c ON c.category_id = lr.category_id
GROUP BY c.category_id, c.category_name
HAVING AVG(lr.rev) > (SELECT AVG(rev) FROM line_rev)
ORDER BY avg_cat_line_rev DESC;

-- Q15: Join + conditional join key - payment rows mapped to order month summary.
WITH month_sales AS (
  SELECT DATE_FORMAT(order_date, '%Y-%m-01') AS month_start, SUM(total_amount) AS gross_sales
  FROM orders
  GROUP BY DATE_FORMAT(order_date, '%Y-%m-01')
)
SELECT
  p.payment_id,
  p.order_id,
  DATE_FORMAT(p.payment_date, '%Y-%m-01') AS pay_month,
  ms.gross_sales
FROM payments p
LEFT JOIN month_sales ms
  ON ms.month_start = DATE_FORMAT(p.payment_date, '%Y-%m-01');
