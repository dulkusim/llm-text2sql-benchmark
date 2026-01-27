import json
import random
import os

OUTPUT_FILE = os.path.join("data", "custom.json")

# --- Templates ---

# EASY: Single table, simple logic
EASY_TEMPLATES = [
    {
        "q": "How many customers live in {city}?",
        "sql": "SELECT count(*) FROM customers WHERE city = '{city}'"
    },
    {
        "q": "List all products in the {category} category.",
        "sql": "SELECT product_name FROM products WHERE category = '{category}'"
    },
    {
        "q": "What is the price of the product with ID {id}?",
        "sql": "SELECT price FROM products WHERE product_id = {id}"
    }
]

# MEDIUM: Exactly 1 JOIN or GROUP BY (Logic: Join=1 or GroupBy)
MEDIUM_TEMPLATES = [
    {
        "q": "Find the total number of orders placed by {fname} {lname}.",
        "sql": "SELECT count(*) FROM orders T1 JOIN customers T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = '{fname}' AND T2.last_name = '{lname}'"
    },
    {
        "q": "What is the average price of products in each category?",
        "sql": "SELECT category, avg(price) FROM products GROUP BY category"
    },
    {
        "q": "List the total revenue generated from each customer city.",
        "sql": "SELECT T2.city, sum(T1.total_amount) FROM orders T1 JOIN customers T2 ON T1.customer_id = T2.customer_id GROUP BY T2.city"
    }
]

# HARD: Exactly 3 JOINS (4 Tables) -> Guaranteed "Hard"
HARD_TEMPLATES = [
    {
        "q": "Find the first name of customers who bought '{product}' but live in '{city}'.",
        "sql": "SELECT T1.first_name FROM customers T1 JOIN orders T2 ON T1.customer_id = T2.customer_id JOIN order_items T3 ON T2.order_id = T3.order_id JOIN products T4 ON T3.product_id = T4.product_id WHERE T4.product_name = '{product}' AND T1.city = '{city}'"
    },
    {
        "q": "List the total quantity of '{product}' ordered by customers in '{city}'.",
        "sql": "SELECT sum(T3.quantity) FROM customers T1 JOIN orders T2 ON T1.customer_id = T2.customer_id JOIN order_items T3 ON T2.order_id = T3.order_id JOIN products T4 ON T3.product_id = T4.product_id WHERE T4.product_name = '{product}' AND T1.city = '{city}'"
    },
    {
        "q": "Which customers have bought '{product}'?",
        "sql": "SELECT DISTINCT T1.first_name, T1.last_name FROM customers T1 JOIN orders T2 ON T1.customer_id = T2.customer_id JOIN order_items T3 ON T2.order_id = T3.order_id JOIN products T4 ON T3.product_id = T4.product_id WHERE T4.product_name = '{product}'"
    }
]

# EXTRA HARD: Contains Subquery (SELECT ... SELECT) -> Guaranteed "Extra Hard"
EXTRA_HARD_TEMPLATES = [
    {
        "q": "Find the customers who have placed an order with a value higher than the average order value.",
        "sql": "SELECT first_name, last_name FROM customers WHERE customer_id IN (SELECT customer_id FROM orders WHERE total_amount > (SELECT avg(total_amount) FROM orders))"
    },
    {
        "q": "List products that have never been ordered.",
        "sql": "SELECT product_name FROM products WHERE product_id NOT IN (SELECT product_id FROM order_items)"
    },
    {
        "q": "Find the order with the highest total amount.",
        "sql": "SELECT order_id FROM orders WHERE total_amount = (SELECT max(total_amount) FROM orders)"
    }
]

def generate_dataset():
    data = []

    # Variables
    cities = ["New York", "London", "Paris", "Tokyo"]
    categories = ["Electronics", "Clothing", "Books", "Home"]
    products = ["Product 1", "Product 5", "Product 10"]

    def fill_template(templates, count, difficulty):
        for _ in range(count):
            t = random.choice(templates)
            q_text = t["q"]
            sql_text = t["sql"]

            # Simple replacements
            if "{city}" in q_text:
                val = random.choice(cities)
                q_text = q_text.replace("{city}", val)
                sql_text = sql_text.replace("{city}", val)
            if "{category}" in q_text:
                val = random.choice(categories)
                q_text = q_text.replace("{category}", val)
                sql_text = sql_text.replace("{category}", val)
            if "{id}" in q_text:
                val = str(random.randint(1, 10))
                q_text = q_text.replace("{id}", val)
                sql_text = sql_text.replace("{id}", val)
            if "{qty}" in q_text:
                val = str(random.randint(5, 50))
                q_text = q_text.replace("{qty}", val)
                sql_text = sql_text.replace("{qty}", val)
            if "{product}" in q_text:
                val = random.choice(products)
                q_text = q_text.replace("{product}", val)
                sql_text = sql_text.replace("{product}", val)
            if "{fname}" in q_text:
                q_text = q_text.replace("{fname}", "User1").replace("{lname}", "Test1")
                sql_text = sql_text.replace("{fname}", "User1").replace("{lname}", "Test1")

            data.append({
                "db_id": "custom",
                "question": q_text,
                "query": sql_text,
                "difficulty_manual_tag": difficulty
            })

    # Generate exact counts
    fill_template(EASY_TEMPLATES, 25, "Easy")
    fill_template(MEDIUM_TEMPLATES, 25, "Medium")
    fill_template(HARD_TEMPLATES, 25, "Hard")
    fill_template(EXTRA_HARD_TEMPLATES, 25, "Extra Hard")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Generated {len(data)} questions in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
