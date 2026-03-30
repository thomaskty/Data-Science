-- ============================================================
-- MySQL 8+ Practice Schema + Data
-- Focus: joins, group by, filters, conditions, window functions
-- ============================================================

DROP SCHEMA IF EXISTS practice;
CREATE SCHEMA practice;
USE practice;

-- Compatibility note:
-- This script avoids recursive CTE inserts so it works on broader MySQL setups.

-- ------------------------------------------------------------
-- 1) Core master tables
-- ------------------------------------------------------------

CREATE TABLE departments (
  department_id INT PRIMARY KEY AUTO_INCREMENT,
  department_name VARCHAR(80) NOT NULL UNIQUE,
  location VARCHAR(80) NOT NULL
);

CREATE TABLE employees (
  employee_id INT PRIMARY KEY AUTO_INCREMENT,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  department_id INT NOT NULL,
  manager_id INT NULL,
  hire_date DATE NOT NULL,
  base_salary DECIMAL(10,2) NOT NULL,
  bonus_pct DECIMAL(5,2) NOT NULL DEFAULT 0.00,
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  CONSTRAINT fk_emp_dept FOREIGN KEY (department_id) REFERENCES departments(department_id),
  CONSTRAINT fk_emp_mgr FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
);

CREATE TABLE customers (
  customer_id INT PRIMARY KEY AUTO_INCREMENT,
  full_name VARCHAR(120) NOT NULL,
  email VARCHAR(150) NOT NULL UNIQUE,
  phone VARCHAR(25) NULL,
  city VARCHAR(80) NOT NULL,
  state_code CHAR(2) NOT NULL,
  signup_date DATE NOT NULL,
  customer_segment ENUM('Bronze','Silver','Gold','Platinum') NOT NULL,
  is_active TINYINT(1) NOT NULL DEFAULT 1
);

CREATE TABLE categories (
  category_id INT PRIMARY KEY AUTO_INCREMENT,
  category_name VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE products (
  product_id INT PRIMARY KEY AUTO_INCREMENT,
  sku VARCHAR(40) NOT NULL UNIQUE,
  product_name VARCHAR(150) NOT NULL,
  category_id INT NOT NULL,
  unit_price DECIMAL(10,2) NOT NULL,
  cost_price DECIMAL(10,2) NOT NULL,
  is_discontinued TINYINT(1) NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL,
  CONSTRAINT fk_prod_cat FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- ------------------------------------------------------------
-- 2) Transaction tables
-- ------------------------------------------------------------

CREATE TABLE orders (
  order_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  customer_id INT NOT NULL,
  employee_id INT NOT NULL,
  order_date DATETIME NOT NULL,
  order_status ENUM('placed','paid','shipped','delivered','cancelled','returned') NOT NULL,
  sales_channel ENUM('web','mobile_app','store','partner') NOT NULL,
  shipping_city VARCHAR(80) NOT NULL,
  total_amount DECIMAL(12,2) NOT NULL DEFAULT 0.00,
  CONSTRAINT fk_order_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  CONSTRAINT fk_order_employee FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE order_items (
  order_item_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  order_id BIGINT NOT NULL,
  product_id INT NOT NULL,
  quantity INT NOT NULL,
  unit_price DECIMAL(10,2) NOT NULL,
  discount_pct DECIMAL(5,2) NOT NULL DEFAULT 0.00,
  CONSTRAINT fk_item_order FOREIGN KEY (order_id) REFERENCES orders(order_id),
  CONSTRAINT fk_item_product FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE payments (
  payment_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  order_id BIGINT NOT NULL,
  payment_date DATETIME NOT NULL,
  payment_method ENUM('card','upi','net_banking','wallet','cod') NOT NULL,
  payment_status ENUM('success','failed','pending','refunded') NOT NULL,
  amount DECIMAL(12,2) NOT NULL,
  transaction_ref VARCHAR(50) NOT NULL UNIQUE,
  CONSTRAINT fk_pay_order FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- ------------------------------------------------------------
-- 3) Analytics-friendly event tables
-- ------------------------------------------------------------

CREATE TABLE customer_logins (
  login_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  customer_id INT NOT NULL,
  login_ts DATETIME NOT NULL,
  device_type ENUM('android','ios','web') NOT NULL,
  success_flag TINYINT(1) NOT NULL,
  CONSTRAINT fk_login_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE support_tickets (
  ticket_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  customer_id INT NOT NULL,
  created_at DATETIME NOT NULL,
  resolved_at DATETIME NULL,
  priority ENUM('low','medium','high','critical') NOT NULL,
  ticket_status ENUM('open','in_progress','resolved','closed') NOT NULL,
  issue_type ENUM('payment','delivery','quality','returns','account') NOT NULL,
  agent_employee_id INT NULL,
  CONSTRAINT fk_ticket_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  CONSTRAINT fk_ticket_agent FOREIGN KEY (agent_employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE employee_monthly_sales (
  sales_month DATE NOT NULL,
  employee_id INT NOT NULL,
  target_amount DECIMAL(12,2) NOT NULL,
  achieved_amount DECIMAL(12,2) NOT NULL,
  PRIMARY KEY (sales_month, employee_id),
  CONSTRAINT fk_month_sales_emp FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

-- Helper numbers table for deterministic data generation
CREATE TABLE seq_numbers (
  n INT PRIMARY KEY
);

INSERT INTO seq_numbers (n)
SELECT ones.n + tens.n * 10 + hundreds.n * 100 + thousands.n * 1000 + 1 AS n
FROM
  (SELECT 0 AS n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) ones
  CROSS JOIN
  (SELECT 0 AS n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) tens
  CROSS JOIN
  (SELECT 0 AS n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) hundreds
  CROSS JOIN
  (SELECT 0 AS n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) thousands
WHERE ones.n + tens.n * 10 + hundreds.n * 100 + thousands.n * 1000 < 5000
ORDER BY 1;

-- ------------------------------------------------------------
-- 4) Indexes (important for realistic querying)
-- ------------------------------------------------------------

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_employee ON orders(employee_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(order_status);

CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);

CREATE INDEX idx_payments_order ON payments(order_id);
CREATE INDEX idx_payments_status ON payments(payment_status);

CREATE INDEX idx_logins_customer_ts ON customer_logins(customer_id, login_ts);
CREATE INDEX idx_tickets_customer_created ON support_tickets(customer_id, created_at);
CREATE INDEX idx_tickets_status ON support_tickets(ticket_status);

-- ------------------------------------------------------------
-- 5) Seed static dimensions
-- ------------------------------------------------------------

INSERT INTO departments (department_name, location) VALUES
('Sales', 'Bengaluru'),
('Marketing', 'Mumbai'),
('Finance', 'Chennai'),
('Operations', 'Hyderabad'),
('Support', 'Pune'),
('Technology', 'Gurugram');

INSERT INTO categories (category_name) VALUES
('Electronics'),
('Home Appliances'),
('Furniture'),
('Books'),
('Fashion'),
('Sports'),
('Grocery'),
('Beauty'),
('Toys'),
('Stationery');

INSERT INTO employees (first_name, last_name, department_id, manager_id, hire_date, base_salary, bonus_pct, is_active)
SELECT
  CONCAT('Emp', LPAD(n, 2, '0')),
  CONCAT('LN', LPAD(n, 2, '0')),
  1 + (n % 6),
  CASE
    WHEN n <= 6 THEN NULL
    ELSE 1 + (n % 6)
  END,
  DATE_ADD('2018-01-01', INTERVAL (n * 37) DAY),
  35000 + (n * 1750),
  ROUND((n % 15) * 0.5, 2),
  CASE WHEN n % 13 = 0 THEN 0 ELSE 1 END
FROM seq_numbers
WHERE n <= 40
ORDER BY n;

-- ------------------------------------------------------------
-- 7) Generate customers (200 rows)
-- ------------------------------------------------------------

INSERT INTO customers (full_name, email, phone, city, state_code, signup_date, customer_segment, is_active)
SELECT
  CONCAT('Customer_', LPAD(n, 3, '0')),
  CONCAT('customer', LPAD(n, 3, '0'), '@mail.com'),
  CONCAT('+91-98', LPAD(10000000 + n, 8, '0')),
  ELT(1 + (n % 10), 'Bengaluru','Mumbai','Delhi','Chennai','Pune','Hyderabad','Kolkata','Ahmedabad','Jaipur','Lucknow'),
  ELT(1 + (n % 10), 'KA','MH','DL','TN','MH','TS','WB','GJ','RJ','UP'),
  DATE_ADD('2022-01-01', INTERVAL (n * 3) DAY),
  CASE
    WHEN n % 20 = 0 THEN 'Platinum'
    WHEN n % 7 = 0 THEN 'Gold'
    WHEN n % 3 = 0 THEN 'Silver'
    ELSE 'Bronze'
  END,
  CASE WHEN n % 17 = 0 THEN 0 ELSE 1 END
FROM seq_numbers
WHERE n <= 200
ORDER BY n;

-- ------------------------------------------------------------
-- 8) Generate products (90 rows)
-- ------------------------------------------------------------

INSERT INTO products (sku, product_name, category_id, unit_price, cost_price, is_discontinued, created_at)
SELECT
  CONCAT('SKU-', LPAD(n, 4, '0')),
  CONCAT('Product_', LPAD(n, 3, '0')),
  1 + (n % 10),
  ROUND(100 + (n * 17.35), 2),
  ROUND((100 + (n * 17.35)) * (0.55 + ((n % 10) / 100)), 2),
  CASE WHEN n % 29 = 0 THEN 1 ELSE 0 END,
  DATE_ADD('2021-01-01 09:00:00', INTERVAL (n * 11) DAY)
FROM seq_numbers
WHERE n <= 90
ORDER BY n;

-- ------------------------------------------------------------
-- 9) Generate orders (1500 rows)
-- ------------------------------------------------------------

INSERT INTO orders (customer_id, employee_id, order_date, order_status, sales_channel, shipping_city, total_amount)
SELECT
  1 + ((n * 37) % 200),
  1 + ((n * 11) % 40),
  TIMESTAMP(
    DATE_ADD('2024-01-01', INTERVAL ((n * 2) % 700) DAY),
    MAKETIME((n * 3) % 24, (n * 7) % 60, 0)
  ),
  CASE
    WHEN n % 23 = 0 THEN 'cancelled'
    WHEN n % 19 = 0 THEN 'returned'
    WHEN n % 5 = 0 THEN 'delivered'
    WHEN n % 4 = 0 THEN 'shipped'
    WHEN n % 3 = 0 THEN 'paid'
    ELSE 'placed'
  END,
  ELT(1 + (n % 4), 'web', 'mobile_app', 'store', 'partner'),
  ELT(1 + (n % 10), 'Bengaluru','Mumbai','Delhi','Chennai','Pune','Hyderabad','Kolkata','Ahmedabad','Jaipur','Lucknow'),
  0.00
FROM seq_numbers
WHERE n <= 1500
ORDER BY n;

-- ------------------------------------------------------------
-- 10) Generate order_items (1-4 items/order)
-- ------------------------------------------------------------

INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount_pct)
SELECT
  o.order_id,
  p.product_id,
  1 + ((o.order_id + i.item_no) % 5),
  p.unit_price,
  CASE
    WHEN (o.order_id + i.item_no) % 11 = 0 THEN 20.00
    WHEN (o.order_id + i.item_no) % 7 = 0 THEN 10.00
    WHEN (o.order_id + i.item_no) % 5 = 0 THEN 5.00
    ELSE 0.00
  END AS discount_pct
FROM orders o
JOIN (
  SELECT 1 AS item_no
  UNION ALL SELECT 2
  UNION ALL SELECT 3
  UNION ALL SELECT 4
) i
  ON i.item_no <= 1 + (o.order_id % 4)
JOIN products p
  ON p.product_id = 1 + ((o.order_id * 13 + i.item_no * 7) % 90);

-- Update order totals from order_items
UPDATE orders o
JOIN (
  SELECT
    oi.order_id,
    ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)), 2) AS calc_total
  FROM order_items oi
  GROUP BY oi.order_id
) t ON t.order_id = o.order_id
SET o.total_amount = t.calc_total;

-- ------------------------------------------------------------
-- 11) Generate payments
-- ------------------------------------------------------------

-- Primary payment row for each order
INSERT INTO payments (order_id, payment_date, payment_method, payment_status, amount, transaction_ref)
SELECT
  o.order_id,
  DATE_ADD(o.order_date, INTERVAL (o.order_id % 72) HOUR),
  ELT(1 + (o.order_id % 5), 'card','upi','net_banking','wallet','cod'),
  CASE
    WHEN o.order_status = 'cancelled' THEN 'failed'
    WHEN o.order_status = 'placed' THEN 'pending'
    WHEN o.order_status = 'returned' THEN 'refunded'
    ELSE 'success'
  END,
  CASE
    WHEN o.order_status = 'cancelled' THEN 0.00
    WHEN o.order_status = 'returned' THEN o.total_amount
    ELSE o.total_amount
  END,
  CONCAT('TXN-', LPAD(o.order_id, 8, '0'), '-A')
FROM orders o;

-- Additional split payment for some successful orders (for grouping/window practice)
INSERT INTO payments (order_id, payment_date, payment_method, payment_status, amount, transaction_ref)
SELECT
  o.order_id,
  DATE_ADD(o.order_date, INTERVAL ((o.order_id % 24) + 1) HOUR),
  ELT(1 + ((o.order_id + 2) % 5), 'card','upi','net_banking','wallet','cod'),
  'success',
  ROUND(o.total_amount * 0.25, 2),
  CONCAT('TXN-', LPAD(o.order_id, 8, '0'), '-B')
FROM orders o
WHERE o.order_id % 9 = 0
  AND o.order_status IN ('paid', 'shipped', 'delivered');

INSERT INTO customer_logins (customer_id, login_ts, device_type, success_flag)
SELECT
  1 + ((n * 17) % 200),
  TIMESTAMP(
    DATE_ADD('2024-01-01', INTERVAL (n % 730) DAY),
    MAKETIME((n * 5) % 24, (n * 9) % 60, (n * 7) % 60)
  ),
  ELT(1 + (n % 3), 'android', 'ios', 'web'),
  CASE WHEN n % 13 = 0 THEN 0 ELSE 1 END
FROM seq_numbers
WHERE n <= 3000
ORDER BY n;

-- ------------------------------------------------------------
-- 13) Generate support_tickets (800 rows)
-- ------------------------------------------------------------

INSERT INTO support_tickets (customer_id, created_at, resolved_at, priority, ticket_status, issue_type, agent_employee_id)
SELECT
  1 + ((n * 19) % 200),
  TIMESTAMP(
    DATE_ADD('2024-01-01', INTERVAL (n % 700) DAY),
    MAKETIME((n * 2) % 24, (n * 13) % 60, 0)
  ),
  CASE
    WHEN n % 6 = 0 THEN NULL
    ELSE DATE_ADD(
      TIMESTAMP(DATE_ADD('2024-01-01', INTERVAL (n % 700) DAY), MAKETIME((n * 2) % 24, (n * 13) % 60, 0)),
      INTERVAL (2 + (n % 120)) HOUR
    )
  END,
  ELT(1 + (n % 4), 'low', 'medium', 'high', 'critical'),
  CASE
    WHEN n % 6 = 0 THEN 'open'
    WHEN n % 5 = 0 THEN 'in_progress'
    WHEN n % 2 = 0 THEN 'resolved'
    ELSE 'closed'
  END,
  ELT(1 + (n % 5), 'payment','delivery','quality','returns','account'),
  1 + ((n * 7) % 40)
FROM seq_numbers
WHERE n <= 800
ORDER BY n;

-- ------------------------------------------------------------
-- 14) Generate employee_monthly_sales (24 months x 40 employees)
-- ------------------------------------------------------------

INSERT INTO employee_monthly_sales (sales_month, employee_id, target_amount, achieved_amount)
SELECT
  DATE_ADD('2024-01-01', INTERVAL m.n - 1 MONTH) AS sales_month,
  e.employee_id,
  ROUND(80000 + (e.employee_id * 1200) + ((m.n - 1) * 900), 2) AS target_amount,
  ROUND((80000 + (e.employee_id * 1200) + ((m.n - 1) * 900)) * (0.70 + ((e.employee_id + (m.n - 1)) % 45) / 100), 2) AS achieved_amount
FROM (SELECT n FROM seq_numbers WHERE n <= 24) m
CROSS JOIN employees e
ORDER BY m.n, e.employee_id;

-- ------------------------------------------------------------
-- 15) Quick row-count checks
-- ------------------------------------------------------------

SELECT 'departments' AS table_name, COUNT(*) AS row_count FROM departments
UNION ALL SELECT 'employees', COUNT(*) FROM employees
UNION ALL SELECT 'customers', COUNT(*) FROM customers
UNION ALL SELECT 'categories', COUNT(*) FROM categories
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'payments', COUNT(*) FROM payments
UNION ALL SELECT 'customer_logins', COUNT(*) FROM customer_logins
UNION ALL SELECT 'support_tickets', COUNT(*) FROM support_tickets
UNION ALL SELECT 'employee_monthly_sales', COUNT(*) FROM employee_monthly_sales;

