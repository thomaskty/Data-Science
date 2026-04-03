create database revision;

-- ############# BASIC SELECT STATEMENTS ##################
CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    age INT,
    department VARCHAR(50),
    salary NUMERIC(10,2),
    join_date DATE
);

INSERT INTO employees (first_name, last_name, age, department, salary, join_date) VALUES
('Alice', 'Smith', 28, 'Engineering', 70000, '2021-03-15'),
('Bob', 'Johnson', 35, 'Marketing', 55000, '2019-07-22'),
('Carol', 'Williams', 40, 'Engineering', 95000, '2015-01-12'),
('David', 'Brown', 25, 'Sales', 50000, '2022-11-01'),
('Eve', 'Davis', 30, 'Marketing', 60000, '2020-05-18'),
('Frank', 'Miller', 45, 'Management', 120000, '2010-06-30'),
('Grace', 'Wilson', 32, 'Sales', 52000, '2018-09-10');

SELECT * FROM employees; 
SELECT first_name, last_name, salary FROM employees;

SELECT * FROM employees
WHERE department = 'Engineering';

SELECT * FROM employees
WHERE salary > 60000;

-- AND example
SELECT * FROM employees
WHERE department = 'Engineering' AND salary > 80000;

-- OR example
SELECT * FROM employees
WHERE department = 'Sales' OR department = 'Marketing';

SELECT * FROM employees
ORDER BY salary DESC;

SELECT * FROM employees
ORDER BY salary DESC
LIMIT 3;

-- question1 : List all employees younger than 35.
SELECT * FROM employees
WHERE age<35;

-- question2 : Show only first and last names of Marketing employees.
SELECT first_name,last_name FROM employees
WHERE department='Marketing';

-- question 3 : List employees from Sales or Engineering sorted by join_date ascending.
SELECT * FROM employees
WHERE department='Sales' OR department='Engineering'
ORDER BY join_date ASC; 

-- question 4 : Get the 2 lowest-paid employees.
SELECT * FROM employees
ORDER BY salary ASC
LIMIT 2 ; 

--################ GROUP BY AGGREGATIONS #################
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(50),
    manager VARCHAR(50)
);

INSERT INTO departments (dept_name, manager) VALUES
('Engineering', 'Carol Williams'),
('Marketing', 'Bob Johnson'),
('Sales', 'David Brown'),
('Management', 'Frank Miller');

SELECT COUNT(*) AS total_employees FROM employees;
SELECT AVG(salary) AS avg_salary FROM employees;
SELECT MIN(age) AS youngest, MAX(age) AS oldest FROM employees;

SELECT department, SUM(salary) AS total_salary,AVG(salary) AS average_salary
FROM employees
GROUP BY department;

SELECT department, COUNT(*) AS emp_count
FROM employees
GROUP BY department
HAVING COUNT(*)< 2;

SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 60000;

SELECT department, SUM(salary) AS total_salary
FROM employees
GROUP BY department
ORDER BY total_salary DESC;

-- question 1 : Count how many employees are in each department.
SELECT department,COUNT(DISTINCT(emp_id)) FROM employees GROUP BY department;

-- question 2 : Find the average age of employees per department.
SELECT department,AVG(age) FROM employees GROUP BY department; 

-- question 3 : Show departments having total salary greater than 150000.
SELECT department,SUM(salary) FROM employees GROUP BY department HAVING SUM(salary)>150000; 

-- question 4 : List departments sorted by the number of employees (highest first).
SELECT department,COUNT(DISTINCT(emp_id)) as total_no_employees FROM employees GROUP BY department
ORDER BY COUNT(DISTINCT(emp_id)) DESC ; 

-- question 5 : Find the department(s) with the highest average salary.
SELECT department,AVG(salary) FROM employees GROUP BY department ORDER BY AVG(salary) DESC LIMIT 1 ; 

--######################## JOINS #######################
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(50),
    dept_name VARCHAR(50),
    budget NUMERIC(12,2)
);

INSERT INTO projects (project_name, dept_name, budget) VALUES
('Project Alpha', 'Engineering', 250000),
('Project Beta', 'Engineering', 150000),
('Project Gamma', 'Marketing', 120000),
('Project Delta', 'Sales', 90000),
('Project Epsilon', 'Management', 300000);

SELECT e.first_name, e.last_name, e.department, d.manager
FROM employees e
INNER JOIN departments d
ON e.department = d.dept_name;

-- Employees with their project budgets
SELECT e.first_name, e.last_name, e.department, p.project_name, p.budget
FROM employees e
LEFT JOIN projects p
ON e.department = p.dept_name;

-- Find employees who are older than other employees in the same department
SELECT e1.first_name AS emp1, e2.first_name AS emp2, e1.department, e1.age AS age1, e2.age AS age2
FROM employees e1
JOIN employees e2
ON e1.department = e2.department
AND e1.age > e2.age
ORDER BY e1.department, e1.age DESC;

-- question 1 : List each department and the total budget of its projects (include departments with no projects).
WITH table1 AS (
   SELECT b.dept_name,SUM(b.budget) as total_budget
   FROM projects b
   GROUP BY b.dept_name
)
SELECT a.dept_name,b.total_budget
FROM departments a 
LEFT JOIN table1 b
ON a.dept_name = b.dept_name; 


SELECT a.dept_name,COALESCE(SUM(b.budget),0) as total_budget
FROM departments a
LEFT JOIN projects b 
ON a.dept_name = b.dept_name
GROUP BY a.dept_name; 


-- question 2 : For each employee, list their projects and indicate if the project budget is greater than their salary.
Select e.emp_id,e.salary,e.department,p.budget
from employees e 
left join projects p
on e.department = p.dept_name
where e.salary < p.budget ; 

SELECT e.emp_id, e.salary, e.department, p.project_name, p.budget,
       CASE 
           WHEN p.budget > e.salary THEN 'Budget > Salary'
           ELSE 'Budget <= Salary or No Project'
       END AS budget_vs_salary
FROM employees e
LEFT JOIN projects p
ON e.department = p.dept_name;

-- question 3 : Find employees who work in the same department as 'Alice Smith' (excluding Alice).
SELECT *
FROM employees
WHERE department = (
    SELECT department
    FROM employees
    WHERE first_name = 'Alice' AND last_name = 'Smith'
)
AND NOT (first_name = 'Alice' AND last_name = 'Smith');

SELECT e2.*
FROM employees e1
JOIN employees e2
ON e1.department = e2.department
WHERE e1.first_name = 'Alice'
  AND e1.last_name = 'Smith'
  AND e2.emp_id <> e1.emp_id;


-- question 4 : List departments where the average employee salary exceeds 70000 and total project budget exceeds 200000.
WITH emp_avg AS (
    SELECT department, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
),
proj_total AS (
    SELECT dept_name AS department, SUM(budget) AS total_budget
    FROM projects
    GROUP BY dept_name
)
SELECT e.department, e.avg_salary, p.total_budget
FROM emp_avg e
JOIN proj_total p
ON e.department = p.department
WHERE e.avg_salary > 70000
  AND p.total_budget > 200000;

-- question 5 : Find employees who are older than the average age of their department.
WITH dept_avg AS (
    SELECT department, AVG(age) AS avg_age
    FROM employees
    GROUP BY department
)
SELECT e.emp_id, e.first_name, e.last_name, e.department, e.age, d.avg_age
FROM employees e
JOIN dept_avg d
ON e.department = d.department
WHERE e.age > d.avg_age;


WITH emp_with_avg AS (
    SELECT emp_id, first_name, last_name, department, age,
           AVG(age) OVER (PARTITION BY department) AS avg_age
    FROM employees
)
SELECT *
FROM emp_with_avg
WHERE age > avg_age;


-- ###############################
--         INTERVIEW QUESTIONS 
-- ###############################
-- QUESTION1 : Company wants to find the top-paid employee in each department.
CREATE TABLE emp_q1 (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department VARCHAR(50),
    salary NUMERIC(10,2),
    join_date DATE
);

INSERT INTO emp_q1 (first_name, last_name, department, salary, join_date) VALUES
('Alice','Smith','Engineering',70000,'2021-03-15'),
('Bob','Johnson','Marketing',55000,'2019-07-22'),
('Carol','Williams','Engineering',95000,'2015-01-12'),
('David','Brown','Sales',50000,'2022-11-01'),
('Eve','Davis','Marketing',60000,'2020-05-18'),
('Frank','Miller','Management',120000,'2010-06-30'),
('Grace','Wilson','Sales',52000,'2018-09-10');

SELECT emp_id, first_name, last_name, department, salary, join_date
FROM (
    SELECT *,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
    FROM emp_q1
) sub
WHERE dept_rank = 1;


-- Scenario: Sales company wants to know monthly sales per employee, 
-- but only include employees whose monthly sales exceed the department average.
CREATE TABLE emp_q2 (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department VARCHAR(50)
);

INSERT INTO emp_q2 (first_name, last_name, department) VALUES
('Alice','Smith','Sales'),
('Bob','Johnson','Sales'),
('Carol','Williams','Sales'),
('David','Brown','Marketing'),
('Eve','Davis','Marketing');


CREATE TABLE sales_q2 (
    sale_id SERIAL PRIMARY KEY,
    emp_id INT,
    sale_amount NUMERIC(10,2),
    sale_date DATE
);

INSERT INTO sales_q2 (emp_id, sale_amount, sale_date) VALUES
(1, 1000, '2026-01-01'),
(1, 2000, '2026-01-15'),
(2, 1500, '2026-01-10'),
(3, 2500, '2026-01-20'),
(4, 3000, '2026-01-05'),
(5, 3500, '2026-01-25');

WITH emp_monthly AS (
    SELECT e.emp_id, e.first_name, e.department,
           DATE_TRUNC('month', s.sale_date) AS month,
           SUM(s.sale_amount) AS total_sales
    FROM emp_q2 e
    JOIN sales_q2 s ON e.emp_id = s.emp_id
    GROUP BY e.emp_id, e.first_name, e.department, DATE_TRUNC('month', s.sale_date)
),
dept_avg AS (
    SELECT department, month, AVG(total_sales) AS avg_sales
    FROM emp_monthly
    GROUP BY department, month
)
SELECT em.emp_id, em.first_name, em.department, em.month, em.total_sales
FROM emp_monthly em
JOIN dept_avg da
  ON em.department = da.department AND em.month = da.month
WHERE em.total_sales > da.avg_sales;


-- Scenario: A company tracks employee projects. Find each employee’s rank by salary within their project, 
-- and include only employees whose rank is 1 or 2.
CREATE TABLE emp_q3 (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    salary NUMERIC(10,2)
);

INSERT INTO emp_q3 (first_name, last_name, salary) VALUES
('Alice','Smith',70000),
('Bob','Johnson',55000),
('Carol','Williams',95000),
('David','Brown',50000),
('Eve','Davis',60000);

CREATE TABLE projects_q3 (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(50)
);

INSERT INTO projects_q3 (project_name) VALUES
('Project A'),('Project B');

CREATE TABLE emp_proj_q3 (
    emp_id INT,
    project_id INT
);

INSERT INTO emp_proj_q3 (emp_id, project_id) VALUES
(1,1),(2,1),(3,1),(4,2),(5,2);

SELECT *
FROM (
    SELECT ep.project_id, p.project_name, e.emp_id, e.first_name, e.salary,
           RANK() OVER (PARTITION BY ep.project_id ORDER BY e.salary DESC) AS proj_rank
    FROM emp_proj_q3 ep
    JOIN emp_q3 e ON ep.emp_id = e.emp_id
    JOIN projects_q3 p ON ep.project_id = p.project_id
) sub
WHERE proj_rank <= 2;

-- Scenario: Retail company wants to identify employees whose monthly sales exceed both the department average and the company average, 
-- and list their rank within department and overall.
CREATE TABLE emp_q4 (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    department VARCHAR(50)
);

INSERT INTO emp_q4 (first_name, department) VALUES
('Alice','Sales'),
('Bob','Sales'),
('Carol','Marketing'),
('David','Marketing'),
('Eve','Sales');

CREATE TABLE sales_q4 (
    sale_id SERIAL PRIMARY KEY,
    emp_id INT,
    sale_amount NUMERIC(10,2),
    sale_date DATE
);

INSERT INTO sales_q4 (emp_id, sale_amount, sale_date) VALUES
(1,1000,'2026-01-01'),
(1,2000,'2026-01-15'),
(2,1500,'2026-01-10'),
(3,3000,'2026-01-05'),
(4,3500,'2026-01-12'),
(5,2500,'2026-01-20');

WITH emp_monthly AS (
    SELECT e.emp_id, e.first_name, e.department,
           DATE_TRUNC('month', s.sale_date) AS month,
           SUM(s.sale_amount) AS total_sales
    FROM emp_q4 e
    JOIN sales_q4 s ON e.emp_id = s.emp_id
    GROUP BY e.emp_id, e.first_name, e.department, DATE_TRUNC('month', s.sale_date)
),
dept_avg AS (
    SELECT department, month, AVG(total_sales) AS avg_sales
    FROM emp_monthly
    GROUP BY department, month
),
company_avg AS (
    SELECT month, AVG(total_sales) AS avg_sales
    FROM emp_monthly
    GROUP BY month
)
SELECT emp_id, first_name, department, month, total_sales,
       RANK() OVER (PARTITION BY department, month ORDER BY total_sales DESC) AS dept_rank,
       RANK() OVER (PARTITION BY month ORDER BY total_sales DESC) AS overall_rank
FROM emp_monthly em
JOIN dept_avg da ON em.department = da.department AND em.month = da.month
JOIN company_avg ca ON em.month = ca.month
WHERE em.total_sales > da.avg_sales
  AND em.total_sales > ca.avg_sales
ORDER BY month, department, dept_rank;
