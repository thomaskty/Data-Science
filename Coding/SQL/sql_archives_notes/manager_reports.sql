-- Table: Employee
-- 
-- +-------------+---------+
-- | Column Name | Type    |
-- +-------------+---------+
-- | id          | int     |
-- | name        | varchar |
-- | department  | varchar |
-- | managerId   | int     |
-- +-------------+---------+
-- id is the primary key (column with unique values) for this table.
-- Each row of this table indicates the name of an employee, their department, and the id of their manager.
-- If managerId is null, then the employee does not have a manager.
-- No employee will be the manager of themself.
--  
-- 
-- Write a solution to find managers with at least five direct reports.
-- 
-- Return the result table in any order.
-- 
-- The result format is in the following example.

-- my solution 
with mangerid_count as (
    select a.managerid,count(a.managerid) as report_count
    from employee a
    group by a.managerid
)
select b.name 
from mangerid_count a 
inner join employee b
on a.managerid = b.id
and a.report_count>=5; 

-- another solution 
SELECT name FROM Employee WHERE id IN (
  SELECT managerId FROM Employee GROUP BY managerId HAVING COUNT(*) >= 5
)

-- 
-- Example 1:
-- 
-- Input: 
-- Employee table:
-- +-----+-------+------------+-----------+
-- | id  | name  | department | managerId |
-- +-----+-------+------------+-----------+
-- | 101 | John  | A          | null      |
-- | 102 | Dan   | A          | 101       |
-- | 103 | James | A          | 101       |
-- | 104 | Amy   | A          | 101       |
-- | 105 | Anne  | A          | 101       |
-- | 106 | Ron   | B          | 101       |
-- +-----+-------+------------+-----------+
-- Output: 
-- +------+
-- | name |
-- +------+
-- | John |
-- +------+