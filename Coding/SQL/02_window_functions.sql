

 -- 1. For each customer, assign a row number to their orders from newest to oldest.
 select o.customer_id,o.order_id,order_date,order_status,
 row_number() over(partition by o.customer_id order by order_date desc) as sorted
 from orders o;  

-- 2. For each customer, rank their orders by total_amount (highest first) using both RANK() and DENSE_RANK()
select customer_id,order_id,total_amount,
rank() over(partition by customer_id order by total_amount desc) as rank_value,
dense_rank() over(partition by customer_id order by total_amount desc) as dense_rank_value
from orders; 

-- 3. Return only the top 3 highest-value orders per customer.
select * from (
	select customer_id,order_id,total_amount,
	dense_rank() over(partition by customer_id order by total_amount desc) as top3
	from orders
)a
where a.top3<=3; 

-- 4. For each customer, compute a running total of total_amount over time
select a.customer_id,a.order_date,a.order_id,a.total_amount,
sum(a.total_amount) over (partition by a.customer_id order by a.order_date asc, a.order_id asc
rows between unbounded preceding and current row) as running_total
from orders a ; 

-- 5. For each order, show previous order amount and next order amount for the same customer.
select a.order_id,a.customer_id,
lag(a.total_amount,1)  over (partition by a.customer_id order by a.order_date asc) as previous_order_amount,
a.total_amount as current_order_amount, 
lead(a.total_amount,1) over (partition by a.customer_id order by a.order_date asc) as next_order_amount
from orders a ; 

-- 6. For each order, compute difference from previous order amount for that customer.
select a.order_id,a.customer_id,a.total_amount,
lag(a.total_amount,1) over (partition by a.customer_id order by a.order_date asc) as prev_order,
(lag(a.total_amount,1) over (partition by a.customer_id order by a.order_date asc) -a.total_amount )as diff 
from orders a ; 

-- 7. For each customer, calculate each order’s contribution percentage to that customer’s total order amount.
select a.customer_id,a.order_id,a.total_amount,
sum(a.total_amount) over (partition by a.customer_id) as total_sum_amount,
(a.total_amount/sum(a.total_amount) over (partition by a.customer_id)) as contribution
from orders a ; 

-- 8. For each sales channel, compute monthly sales and month-over-month (MoM) difference using LAG()
with monthly_sales as (
select a.sales_channel,DATE_FORMAT(a.order_date, '%Y%m') as yyyymm,sum(a.total_amount) as total_sales
from orders a
group by DATE_FORMAT(a.order_date, '%Y%m'),a.sales_channel
order by a.sales_channel asc, DATE_FORMAT(a.order_date, '%Y%m') asc
)
select a.sales_channel,a.yyyymm,a.total_sales as monthly_total_sales,
lag(a.total_sales) over (partition by sales_channel order by a.yyyymm asc) as previous_month_sales,
a.total_sales - lag(a.total_sales) over (partition by sales_channel order by a.yyyymm asc) as improvement
from monthly_sales a ; 

-- 9. For each month, assign percentile rank (PERCENT_RANK()) to orders by total_amount
with month_table as (
select date_format(a.order_date,'%Y-%m') as yyyymm,
a.total_amount, a.order_id
from orders a 
)
select a.yyyymm,a.total_amount,a.order_id,
percent_rank() over (partition by a.yyyymm order by a.total_amount) as percentile_rank
from month_table a
order by a.yyyymm,a.total_amount, a.order_id;


-- 10. For each customer, return only the latest order and flag whether its amount is above that customer’s historical average 
with ordered_data as (
select a.customer_id,a.order_id,a.order_date,a.total_amount,
row_number() over (partition by a.customer_id order by a.order_date desc) as rn,
avg(a.total_amount) over 
	(partition by a.customer_id order by a.order_date desc rows between 1 following and unbounded following ) 
as historical_avg
from orders a
)
select a.customer_id,a.order_id,a.order_date,a.total_amount,a.historical_avg,
case when a.total_amount > a.historical_avg then 1 else 0 end as flag
from ordered_data a 
where a.rn=1; 


-- 10. Identify first and latest order date per customer using window MIN/MAX.
SELECT DISTINCT
  o.customer_id,
  MIN(o.order_date) OVER (PARTITION BY o.customer_id) AS first_order_date,
  MAX(o.order_date) OVER (PARTITION BY o.customer_id) AS latest_order_date
FROM orders o;

-- 11. Calculate a 3-order moving average of order amount per customer.
SELECT
  o.customer_id,
  o.order_id,
  o.order_date,
  o.total_amount,
  AVG(o.total_amount) OVER (
    PARTITION BY o.customer_id
    ORDER BY o.order_date, o.order_id
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS moving_avg_3_orders
FROM orders o;

-- 12. For each product, rank order_items by line revenue (qty * price * discount-adjusted).
SELECT
  oi.product_id,
  oi.order_item_id,
  ROUND(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100), 2) AS line_revenue,
  ROW_NUMBER() OVER (
    PARTITION BY oi.product_id
    ORDER BY (oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC, oi.order_item_id DESC
  ) AS rn
FROM order_items oi;

-- 13. For each department, list employees by salary and show salary percentile bucket (NTILE 4).
SELECT
  e.department_id,
  e.employee_id,
  e.base_salary,
  NTILE(4) OVER (PARTITION BY e.department_id ORDER BY e.base_salary DESC) AS salary_quartile
FROM employees e;

-- 14. Show first and last order amount in each customer timeline using FIRST_VALUE/LAST_VALUE.
SELECT
  o.customer_id,
  o.order_id,
  o.order_date,
  o.total_amount,
  FIRST_VALUE(o.total_amount) OVER (
    PARTITION BY o.customer_id
    ORDER BY o.order_date, o.order_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS first_order_amount,
  LAST_VALUE(o.total_amount) OVER (
    PARTITION BY o.customer_id
    ORDER BY o.order_date, o.order_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS last_order_amount
FROM orders o;


