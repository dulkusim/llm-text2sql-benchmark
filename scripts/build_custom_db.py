import sqlite3
import random
import os
from datetime import date, timedelta

# Setup paths
DB_FOLDER = os.path.join("data", "database", "custom")
os.makedirs(DB_FOLDER, exist_ok=True)
DB_PATH = os.path.join(DB_FOLDER, "custom.sqlite")
SCHEMA_PATH = os.path.join(DB_FOLDER, "schema.sql")

def generate_data():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Load Schema
    with open(SCHEMA_PATH, "r") as f:
        cursor.executescript(f.read())

    print("✅ Schema created.")

    # 2. Add Customers
    cities = ["New York", "London", "Paris", "Tokyo", "Berlin"]
    customers = []
    for i in range(1, 51):  # 50 customers
        fname = f"User{i}"
        lname = f"Test{i}"
        city = random.choice(cities)
        cursor.execute(
            "INSERT INTO customers (customer_id, first_name, last_name, email, city, join_date) VALUES (?, ?, ?, ?, ?, ?)",
            (i, fname, lname, f"user{i}@example.com", city, "2023-01-01")
        )
        customers.append(i)

    # 3. Add Products
    categories = ["Electronics", "Clothing", "Books", "Home"]
    products = []
    for i in range(1, 21): # 20 products
        price = random.randint(10, 1000)
        cursor.execute(
            "INSERT INTO products (product_id, product_name, category, price, stock_quantity) VALUES (?, ?, ?, ?, ?)",
            (i, f"Product {i}", random.choice(categories), price, random.randint(0, 100))
        )
        products.append(i)

    # 4. Add Orders & Items
    for i in range(1, 101): # 100 orders
        cust_id = random.choice(customers)
        status = random.choice(["Shipped", "Pending", "Cancelled"])
        cursor.execute(
            "INSERT INTO orders (order_id, customer_id, order_date, status, total_amount) VALUES (?, ?, ?, ?, ?)",
            (i, cust_id, "2023-06-15", status, 0) # Total calculated later
        )

        # Add 1-3 items per order
        order_total = 0
        for _ in range(random.randint(1, 3)):
            prod_id = random.choice(products)
            qty = random.randint(1, 5)
            # Get price
            price = cursor.execute("SELECT price FROM products WHERE product_id=?", (prod_id,)).fetchone()[0]

            cursor.execute(
                "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)",
                (i, prod_id, qty)
            )
            order_total += price * qty

        # Update order total
        cursor.execute("UPDATE orders SET total_amount=? WHERE order_id=?", (order_total, i))

    conn.commit()
    conn.close()
    print(f"✅ Database populated at: {DB_PATH}")

if __name__ == "__main__":
    generate_data()