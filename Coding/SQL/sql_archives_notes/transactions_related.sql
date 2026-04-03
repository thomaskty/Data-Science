
transaction table: 
cust_id | txn_amount | txn_date
001 | 500 | 01-03-2024
048 | 200 | 15-02-2025
001 | 300 | 20-01-2024
002 | 400 | 10-12-2023
033 | 700 | 05-11-2023
001 | 200 | 25-12-2023
002 | 100 | 30-01-2024
048 | 150 | 12-03-2024
033 | 300 | 18-02-2024
001 | 400 | 22-02-2024
002 | 250 | 14-03-2024
033 | 500 | 28-12-2023

CREATE TABLE transactions (
    cust_id VARCHAR(10),
    txn_amount INT,
    txn_date DATE
); 
INSERT INTO transactions_tbl (cust_id, txn_amount, txn_date) VALUES
('001', 500, STR_TO_DATE('01-03-2024','%d-%m-%Y')),
('048', 200, STR_TO_DATE('15-02-2025','%d-%m-%Y')),
('001', 300, STR_TO_DATE('20-01-2024','%d-%m-%Y')),
('002', 400, STR_TO_DATE('10-12-2023','%d-%m-%Y')),
('033', 700, STR_TO_DATE('05-11-2023','%d-%m-%Y')),
('001', 200, STR_TO_DATE('25-12-2023','%d-%m-%Y')),
('002', 100, STR_TO_DATE('30-01-2024','%d-%m-%Y')),
('048', 150, STR_TO_DATE('12-03-2024','%d-%m-%Y')),
('033', 300, STR_TO_DATE('18-02-2024','%d-%m-%Y')),
('001', 400, STR_TO_DATE('22-02-2024','%d-%m-%Y')),
('002', 250, STR_TO_DATE('14-03-2024','%d-%m-%Y')),
('033', 500, STR_TO_DATE('28-12-2023','%d-%m-%Y'));

WITH base AS (
    SELECT 
        a.*,
        MAX(a.txn_date) OVER (PARTITION BY cust_id) AS max_txn_date,
        CAST(DATE_FORMAT(MAX(a.txn_date) OVER (PARTITION BY cust_id), '%Y%m') AS UNSIGNED) AS max_yyyymm,
        CAST(DATE_FORMAT(a.txn_date, '%Y%m') AS UNSIGNED) AS txn_yyyymm
    FROM transactions_tbl a
)
SELECT 
    cust_id,
    SUM(CASE WHEN max_yyyymm - txn_yyyymm = 0 THEN txn_amount ELSE 0 END) AS txn_sum_l0,
    SUM(CASE WHEN max_yyyymm - txn_yyyymm = 1 THEN txn_amount ELSE 0 END) AS txn_sum_l1,
    SUM(CASE WHEN max_yyyymm - txn_yyyymm = 2 THEN txn_amount ELSE 0 END) AS txn_sum_l2
FROM base
GROUP BY cust_id;


202412-202301


