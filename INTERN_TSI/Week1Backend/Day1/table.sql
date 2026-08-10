CREATE DATABASE mydb;

CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INT
);

INSERT INTO students (name, age)
VALUES ('Dinesh', 20);

SELECT * FROM students;