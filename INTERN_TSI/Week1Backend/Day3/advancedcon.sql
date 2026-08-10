-- ============================================================
-- DAY 3: Advanced PostgreSQL
-- Topics:
-- 1. Subqueries
-- 2. CTEs (WITH)
-- 3. Window Functions
-- 4. JSONB Queries
-- ============================================================

-- ============================================================
-- 1. SUBQUERIES
-- ============================================================

-- Find employees earning more than the average salary

SELECT name, salary
FROM employees
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
);

-- Find employees who belong to the AI department

SELECT name
FROM employees
WHERE department_id = (
    SELECT department_id
    FROM departments
    WHERE department_name = 'AI'
);


-- ============================================================
-- 2. CTE - WITH CLAUSE
-- ============================================================

-- Calculate average salary first,
-- then find employees above average

WITH average_salary AS (
    SELECT AVG(salary) AS avg_salary
    FROM employees
)
SELECT e.name, e.salary
FROM employees e
CROSS JOIN average_salary a
WHERE e.salary > a.avg_salary;


-- CTE with GROUP BY

WITH department_salary AS (
    SELECT
        department_id,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT *
FROM department_salary
WHERE avg_salary > 35000;


-- ============================================================
-- 3. WINDOW FUNCTIONS
-- ============================================================

-- ROW_NUMBER
-- Gives a unique sequential number to each row

SELECT
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num
FROM employees;


-- RANK
-- Employees with the same salary receive the same rank

SELECT
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS salary_rank
FROM employees;


-- ROW_NUMBER by department

SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER (
        PARTITION BY department_id
        ORDER BY salary DESC
    ) AS dept_rank
FROM employees;


-- RANK by department

SELECT
    name,
    department_id,
    salary,
    RANK() OVER (
        PARTITION BY department_id
        ORDER BY salary DESC
    ) AS dept_rank
FROM employees;


-- LAG
-- Gets the value from the previous row

SELECT
    name,
    salary,
    LAG(salary) OVER (
        ORDER BY salary
    ) AS previous_salary
FROM employees;


-- Calculate salary difference from previous employee

SELECT
    name,
    salary,
    LAG(salary) OVER (
        ORDER BY salary
    ) AS previous_salary,
    salary - LAG(salary) OVER (
        ORDER BY salary
    ) AS salary_difference
FROM employees;


-- ============================================================
-- 4. JSONB
-- ============================================================

-- Example table for JSONB

CREATE TABLE employee_details (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    details JSONB
);


-- Insert JSON data

INSERT INTO employee_details (name, details)
VALUES
(
    'Dinesh',
    '{"skills": ["Python", "SQL", "RAG"], "experience": 1, "city": "Chennai"}'
),
(
    'Arun',
    '{"skills": ["Java", "SQL"], "experience": 2, "city": "Bangalore"}'
);


-- View JSONB data

SELECT *
FROM employee_details;


-- Get the complete JSON

SELECT details
FROM employee_details;


-- Get a JSON field as JSON

SELECT details -> 'city'
FROM employee_details;


-- Get a JSON field as text

SELECT details ->> 'city'
FROM employee_details;


-- Find employees from Chennai

SELECT *
FROM employee_details
WHERE details ->> 'city' = 'Chennai';


-- Get experience

SELECT
    name,
    details ->> 'experience' AS experience
FROM employee_details;


-- Check whether JSON contains a value

SELECT *
FROM employee_details
WHERE details @> '{"city": "Chennai"}';


-- Check whether skills contain Python

SELECT *
FROM employee_details
WHERE details @> '{"skills": ["Python"]}';


-- Get the first skill from the skills array

SELECT
    name,
    details -> 'skills' ->> 0 AS first_skill
FROM employee_details;


SELECT *
FROM employees
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
);

WITH avg_salary AS (
    SELECT AVG(salary) AS avg
    FROM employees
)
SELECT *
FROM employees, avg_salary
WHERE salary > avg;