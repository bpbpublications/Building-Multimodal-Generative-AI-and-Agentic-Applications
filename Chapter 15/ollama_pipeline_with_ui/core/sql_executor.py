import logging
import traceback
from core.sqlite_multi_reader import query_all_dbs

def execute_sql(sql_query):
    """
    Federated query execution over multiple SQLite databases.
    Returns combined rows from all instances with DB origin.
    """
    try:
        return query_all_dbs(sql_query)
    except Exception as e:
        logging.error("Federated SQL execution failed")
        logging.error(traceback.format_exc())
        return None

