CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(100),
    salary INTEGER
);

INSERT INTO employees (name, department, salary) VALUES
('Alice', 'Engineering', 5000),
('Bob', 'HR', 3500),
('Charlie', 'Finance', 4000);