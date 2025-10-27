import sqlite3
import os
import random

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Sample realistic names and cities
FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia"]
LAST_NAMES = ["Smith", "Johnson", "Lee", "Brown", "Garcia", "Martinez", "Davis", "Miller", "Wilson", "Taylor"]
CITIES = ["New York", "San Francisco", "Chicago", "Los Angeles", "Austin", "Boston", "Seattle", "Denver", "Miami", "Phoenix"]

def generate_customers(count):
    return [
        (
            f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            random.randint(22, 65),
            random.choice(CITIES)
        )
        for _ in range(count)
    ]

def seed_db(path, data):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS customers")  # Wipe old table
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            city TEXT
        )
    """)
    cursor.executemany("INSERT INTO customers (name, age, city) VALUES (?, ?, ?)", data)
    conn.commit()
    conn.close()

# Split 60 rows: 30 per DB
customers_db1 = generate_customers(30)
customers_db2 = generate_customers(30)

# Write to DBs
seed_db("data/sqlite1.db", customers_db1)
seed_db("data/sqlite2.db", customers_db2)

print("✅ sqlite1.db and sqlite2.db created with realistic sample data (30 rows each).")
