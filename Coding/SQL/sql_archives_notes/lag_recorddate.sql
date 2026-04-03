-- Table: Weather
-- 
-- +---------------+---------+
-- | Column Name   | Type    |
-- +---------------+---------+
-- | id            | int     |
-- | recordDate    | date    |
-- | temperature   | int     |
-- +---------------+---------+
-- id is the column with unique values for this table.
-- There are no different rows with the same recordDate.
-- This table contains information about the temperature on a certain day.
--  
-- 
-- Write a solution to find all dates' Id with higher temperatures compared to its previous dates (yesterday).
-- 
-- Return the result table in any order.
-- 
-- The result format is in the following example.
-- 
--  
-- 
-- Example 1:
-- 
-- Input: 
-- Weather table:
-- +----+------------+-------------+
-- | id | recordDate | temperature |
-- +----+------------+-------------+
-- | 1  | 2015-01-01 | 10          |
-- | 2  | 2015-01-02 | 25          |
-- | 3  | 2015-01-03 | 20          |
-- | 4  | 2015-01-04 | 30          |
-- +----+------------+-------------+
-- Output: 
-- +----+
-- | id |
-- +----+
-- | 2  |
-- | 4  |
-- +----+
-- Explanation: 
-- In 2015-01-02, the temperature was higher than the previous day (10 -> 25).
-- In 2015-01-04, the temperature was higher than the previous day (20 -> 30).

select a.id from (
select a.id,a.temperature,a.recorddate,
lag(a.temperature,1) over (order by recorddate asc) as previous_temperature,
lag(a.recorddate,1) over(order by recorddate asc) as previous_recorddate
from weather a 
) a 
where a.temperature > a.previous_temperature  
and datediff(previous_recorddate,recorddate) = -1 ; 


-- another solution 
select w1.id from weather w1 
join weather w2 
on datediff(w1.recordDate, w2.recordDate) = 1
where w1.temperature > w2.temperature


-- another solution
select w1.id from Weather w1,Weather w2
where w1.temperature>w2.temperature and datediff(w1.recordDate,w2.recordDate)=1


--
SELECT w2.id from Weather w1 JOIN Weather w2
ON w2.temperature > w1.temperature AND
datediff(w2.recordDate,w1.recordDate) = 1;


