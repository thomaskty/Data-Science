-- ============================================================
-- MySQL 8+ String Functions Interview Practice (Varied)
-- Schema: practice
-- ============================================================

USE practice;

-- Q1: Build username using lower, replace, and concat.
SELECT
  customer_id,
  full_name,
  CONCAT(LOWER(REPLACE(full_name, ' ', '')), '_', customer_id) AS username
FROM customers;

-- Q2: Extract email local part and domain.
SELECT
  customer_id,
  email,
  SUBSTRING_INDEX(email, '@', 1) AS local_part,
  SUBSTRING_INDEX(email, '@', -1) AS domain
FROM customers;

-- Q3: Pad product id with leading zeros.
SELECT
  product_id,
  LPAD(product_id, 6, '0') AS padded_product_id
FROM products;

-- Q4: Trim and normalize city text.
SELECT
  city,
  UPPER(TRIM(city)) AS city_norm
FROM customers;

-- Q5: Find position of first underscore in product_name.
SELECT
  product_id,
  product_name,
  LOCATE('_', product_name) AS underscore_pos
FROM products;

-- Q6: Substring extraction - first 7 chars of SKU.
SELECT
  product_id,
  sku,
  SUBSTRING(sku, 1, 7) AS sku_prefix
FROM products;

-- Q7: Replace underscores with spaces and title-like casing.
SELECT
  product_id,
  product_name,
  CONCAT(UCASE(LEFT(REPLACE(product_name, '_', ' '), 1)), LCASE(SUBSTRING(REPLACE(product_name, '_', ' '), 2))) AS pretty_name
FROM products;

-- Q8: Reverse string for debugging/transformation exercises.
SELECT product_id, product_name, REVERSE(product_name) AS reversed_name
FROM products;

-- Q9: Regex filter - emails ending in mail.com.
SELECT customer_id, email
FROM customers
WHERE email REGEXP 'mail\\.com$';

-- Q10: Regex replace - normalize multiple spaces to single space.
SELECT
  customer_id,
  REGEXP_REPLACE(CONCAT(full_name, '   sample'), ' +', ' ') AS normalized_text
FROM customers
LIMIT 20;

-- Q11: Concatenate hierarchical ticket label.
SELECT
  ticket_id,
  CONCAT('TKT-', LPAD(ticket_id, 6, '0'), '-', UPPER(priority)) AS ticket_label
FROM support_tickets;

-- Q12: CASE-insensitive search in order_status.
SELECT order_id, order_status
FROM orders
WHERE LOWER(order_status) LIKE '%ed%';

-- Q13: Compare CHAR_LENGTH vs LENGTH for ASCII strings.
SELECT
  product_id,
  product_name,
  CHAR_LENGTH(product_name) AS char_len,
  LENGTH(product_name) AS byte_len
FROM products;

-- Q14: Group by email domain.
SELECT
  SUBSTRING_INDEX(email, '@', -1) AS domain,
  COUNT(*) AS cnt
FROM customers
GROUP BY domain
ORDER BY cnt DESC;

-- Q15: Build fixed-width export line for employee summary.
SELECT
  employee_id,
  CONCAT(
    RPAD(CONCAT(first_name, ' ', last_name), 20, ' '),
    LPAD(CAST(base_salary AS CHAR), 10, '0'),
    RPAD(CAST(department_id AS CHAR), 3, ' ')
  ) AS fixed_width_line
FROM employees;
