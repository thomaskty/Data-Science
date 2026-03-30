-- ============================================================
-- MySQL 8+ CASE / Conditional Logic Interview Practice (Varied)
-- Schema: practice
-- ============================================================

USE practice;

-- Q1: Searched CASE - bucket order value.
SELECT
  order_id,
  total_amount,
  CASE
    WHEN total_amount >= 6000 THEN 'vip'
    WHEN total_amount >= 3000 THEN 'high'
    WHEN total_amount >= 1000 THEN 'mid'
    ELSE 'low'
  END AS order_band
FROM orders;

-- Q2: Simple CASE - map order_status to lifecycle stage.
SELECT
  order_id,
  order_status,
  CASE order_status
    WHEN 'placed' THEN 'pre_fulfillment'
    WHEN 'paid' THEN 'pre_fulfillment'
    WHEN 'shipped' THEN 'in_transit'
    WHEN 'delivered' THEN 'fulfilled'
    WHEN 'returned' THEN 'post_fulfillment'
    WHEN 'cancelled' THEN 'closed_lost'
    ELSE 'unknown'
  END AS lifecycle_stage
FROM orders;

-- Q3: CASE in ORDER BY - prioritize unresolved high priority tickets first.
SELECT ticket_id, priority, ticket_status, created_at
FROM support_tickets
ORDER BY
  CASE WHEN ticket_status IN ('open', 'in_progress') THEN 0 ELSE 1 END,
  CASE priority WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END,
  created_at;

-- Q4: CASE + conditional sum - successful/failed payment totals.
SELECT
  SUM(CASE WHEN payment_status = 'success' THEN amount ELSE 0 END) AS success_amount,
  SUM(CASE WHEN payment_status = 'failed' THEN amount ELSE 0 END) AS failed_amount,
  SUM(CASE WHEN payment_status = 'refunded' THEN amount ELSE 0 END) AS refunded_amount
FROM payments;

-- Q5: IF() function - compact binary label for active customers.
SELECT
  customer_id,
  full_name,
  IF(is_active = 1, 'active', 'inactive') AS active_label
FROM customers;

-- Q6: NULLIF() - avoid divide-by-zero in achievement ratio.
SELECT
  employee_id,
  sales_month,
  achieved_amount,
  target_amount,
  ROUND(achieved_amount / NULLIF(target_amount, 0), 4) AS achievement_ratio
FROM employee_monthly_sales;

-- Q7: COALESCE() - manager fallback with multi-level preference.
SELECT
  e.employee_id,
  CONCAT(e.first_name, ' ', e.last_name) AS employee_name,
  COALESCE(CONCAT(m.first_name, ' ', m.last_name), CONCAT('DeptHead_', d.department_name), 'NoMapping') AS reporting_label
FROM employees e
LEFT JOIN employees m ON m.employee_id = e.manager_id
LEFT JOIN departments d ON d.department_id = e.department_id;

-- Q8: CASE in JOIN-derived metric - payment coverage classification by order.
SELECT
  o.order_id,
  o.total_amount,
  COALESCE(SUM(CASE WHEN p.payment_status = 'success' THEN p.amount ELSE 0 END), 0) AS paid_success,
  CASE
    WHEN COALESCE(SUM(CASE WHEN p.payment_status = 'success' THEN p.amount ELSE 0 END), 0) = 0 THEN 'unpaid'
    WHEN COALESCE(SUM(CASE WHEN p.payment_status = 'success' THEN p.amount ELSE 0 END), 0) < o.total_amount THEN 'partial'
    WHEN COALESCE(SUM(CASE WHEN p.payment_status = 'success' THEN p.amount ELSE 0 END), 0) = o.total_amount THEN 'full'
    ELSE 'overpaid'
  END AS payment_state
FROM orders o
LEFT JOIN payments p ON p.order_id = o.order_id
GROUP BY o.order_id, o.total_amount;

-- Q9: Nested CASE - ticket SLA by priority-specific thresholds.
SELECT
  ticket_id,
  priority,
  created_at,
  resolved_at,
  CASE
    WHEN resolved_at IS NULL THEN 'pending'
    ELSE CASE
      WHEN priority = 'critical' AND TIMESTAMPDIFF(HOUR, created_at, resolved_at) <= 4 THEN 'met'
      WHEN priority = 'high' AND TIMESTAMPDIFF(HOUR, created_at, resolved_at) <= 12 THEN 'met'
      WHEN priority = 'medium' AND TIMESTAMPDIFF(HOUR, created_at, resolved_at) <= 24 THEN 'met'
      WHEN priority = 'low' AND TIMESTAMPDIFF(HOUR, created_at, resolved_at) <= 48 THEN 'met'
      ELSE 'breached'
    END
  END AS sla_result
FROM support_tickets;

-- Q10: CASE + window result - compare latest order vs historical avg.
WITH x AS (
  SELECT
    o.customer_id,
    o.order_id,
    o.order_date,
    o.total_amount,
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date DESC, o.order_id DESC) AS rn,
    AVG(o.total_amount) OVER (
      PARTITION BY o.customer_id
      ORDER BY o.order_date DESC, o.order_id DESC
      ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
    ) AS historical_avg
  FROM orders o
)
SELECT
  customer_id,
  order_id,
  total_amount,
  historical_avg,
  CASE
    WHEN historical_avg IS NULL THEN 'no_history'
    WHEN total_amount > historical_avg THEN 'above_history'
    ELSE 'not_above_history'
  END AS latest_vs_history
FROM x
WHERE rn = 1;

-- Q11: CASE as boolean flag columns for feature engineering style output.
SELECT
  order_id,
  total_amount,
  (order_status IN ('cancelled', 'returned')) AS is_lost_order,
  (sales_channel IN ('mobile_app', 'web')) AS is_digital_channel,
  (total_amount >= 3000) AS is_large_order
FROM orders;

-- Q12: CASE + string function - normalize city labels.
SELECT
  customer_id,
  city,
  CASE
    WHEN UPPER(city) IN ('BENGALURU', 'MUMBAI', 'DELHI', 'CHENNAI') THEN 'metro'
    ELSE 'non_metro'
  END AS city_tier
FROM customers;
