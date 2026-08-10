-- Day 2: SQL Basics

-- SELECT
SELECT * FROM employees;

-- WHERE
SELECT *
FROM employees
WHERE salary > 35000;

-- ORDER BY
SELECT *
FROM employees
ORDER BY salary DESC;

-- LIMIT
SELECT *
FROM employees
LIMIT 2;

-- COUNT
SELECT COUNT(*)
FROM employees;

-- SUM
SELECT SUM(salary)
FROM employees;

-- AVG
SELECT AVG(salary)
FROM employees;

-- GROUP BY
SELECT department_id, AVG(salary)
FROM employees
GROUP BY department_id;

-- INNER JOIN
SELECT e.name, d.department_name
FROM employees e
INNER JOIN departments d
ON e.department_id = d.department_id;

-- LEFT JOIN
SELECT d.department_name, e.name
FROM departments d
LEFT JOIN employees e
ON d.department_id = e.department_id;

-- RIGHT JOIN
SELECT e.name, d.department_name
FROM employees e
RIGHT JOIN departments d
ON e.department_id = d.department_id;