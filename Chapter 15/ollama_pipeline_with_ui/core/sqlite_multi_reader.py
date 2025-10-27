import sqlite3

DB_PATHS = ["data/sqlite1.db", "data/sqlite2.db"]  # Place your DB files here

DB_CONNECTIONS = {
    "db1": "data/sqlite1.db",
    "db2": "data/sqlite2.db"
}

def query_all_dbs(sql_query):
    results = []
    for db_name, path in DB_CONNECTIONS.items():
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            # Attach DB name to metadata for federation context
            results.extend([(db_name, row) for row in rows])
        except Exception as e:
            results.append((db_name, f"Error executing on {db_name}: {e}"))
        finally:
            conn.close()
    return results