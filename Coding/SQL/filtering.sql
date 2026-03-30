-- ============================================================
-- MySQL 8+ Filtering Interview Practice (Varied Predicates)
-- Schema: practice
-- ============================================================

USE practice;

-- Q1: Compound filters with AND/OR and parentheses.
SELECT order_id, customer_id, order_status, total_amount
FROM orders
WHERE (order_status IN ('paid', 'shipped', 'delivered') AND total_amount >= 2500)
   OR (order_status = 'placed' AND total_amount >= 5000);

-- Q2: Date interval filter using max order date anchor.
SELECT order_id, order_date, total_amount
FROM orders
WHERE order_date >= (SELECT DATE_SUB(MAX(order_date), INTERVAL 60 DAY) FROM orders);

-- Q3: BETWEEN and NOT BETWEEN on numeric range.
SELECT order_id, total_amount
FROM orders
WHERE total_amount BETWEEN 1500 AND 4500
  AND total_amount NOT BETWEEN 2400 AND 2600;

-- Q4: IN with derived set - orders for top 5 spending customers.
SELECT *
FROM orders
WHERE customer_id IN (
  SELECT customer_id
  FROM (
    SELECT customer_id, SUM(total_amount) AS spend
    FROM orders
    GROUP BY customer_id
    ORDER BY spend DESC
    LIMIT 5
  ) t
);

-- Q5: EXISTS - customers with at least one unresolved ticket.
SELECT c.customer_id, c.full_name
FROM customers c
WHERE EXISTS (
  SELECT 1
  FROM support_tickets t
  WHERE t.customer_id = c.customer_id
    AND t.ticket_status IN ('open', 'in_progress')
);

-- Q6: NOT EXISTS - products never sold with discount.
SELECT p.product_id, p.product_name
FROM products p
WHERE NOT EXISTS (
  SELECT 1
  FROM order_items oi
  WHERE oi.product_id = p.product_id
    AND oi.discount_pct > 0
);

-- Q7: HAVING filter after grouping - channels with avg order amount above global avg.
SELECT sales_channel, AVG(total_amount) AS channel_avg
FROM orders
GROUP BY sales_channel
HAVING AVG(total_amount) > (SELECT AVG(total_amount) FROM orders);

-- Q8: REGEXP filter - customers with numeric suffix 0 or 5 in full_name.
SELECT customer_id, full_name
FROM customers
WHERE full_name REGEXP '_(.*[05])$';

-- Q9: LIKE escape example - match literal underscore in product_name.
SELECT product_id, product_name
FROM products
WHERE product_name LIKE 'Product\_%' ESCAPE '\\';

-- Q10: Window-filter pattern - keep latest order per customer only.
WITH ranked AS (
  SELECT
    o.*,
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date DESC, o.order_id DESC) AS rn
  FROM orders o
)
SELECT *
FROM ranked
WHERE rn = 1;

-- Q11: Filtering with ANY/SOME - orders greater than ANY cancelled-order amount.
SELECT order_id, total_amount
FROM orders
WHERE total_amount > ANY (
  SELECT total_amount
  FROM orders
  WHERE order_status = 'cancelled'
);

-- Q12: Filtering with ALL - orders greater than ALL returned-order amounts.
SELECT order_id, total_amount
FROM orders
WHERE total_amount > ALL (
  SELECT total_amount
  FROM orders
  WHERE order_status = 'returned'
);

-- Q13: Null-sensitive filtering - tickets resolved but missing agent assignment.
SELECT ticket_id, customer_id, resolved_at, agent_employee_id
FROM support_tickets
WHERE resolved_at IS NOT NULL
  AND agent_employee_id IS NULL;

-- Q14: Filter by calculated expression - underpaid orders.
WITH order_paid AS (
  SELECT
    o.order_id,
    o.total_amount,
    COALESCE(SUM(CASE WHEN p.payment_status = 'success' THEN p.amount ELSE 0 END), 0) AS paid_success
  FROM orders o
  LEFT JOIN payments p ON p.order_id = o.order_id
  GROUP BY o.order_id, o.total_amount
)
SELECT *
FROM order_paid
WHERE paid_success < total_amount;

-- Q15: Combined filter with scalar subquery + correlation.
SELECT o.order_id, o.customer_id, o.total_amount
FROM orders o
WHERE o.total_amount > (
  SELECT AVG(o2.total_amount)
  FROM orders o2
  WHERE o2.customer_id = o.customer_id
)
AND o.total_amount > (SELECT AVG(total_amount) FROM orders);
