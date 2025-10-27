import redis
import hashlib
import json
from core.sql_executor import execute_sql

cache = redis.Redis(host='localhost', port=6379, db=0)

def cache_query(sql_query):
    query_hash = hashlib.md5(sql_query.encode()).hexdigest()
    cached_result = cache.get(query_hash)
    if cached_result:
        return json.loads(cached_result)
    result = execute_sql(sql_query)
    if result:
        cache.set(query_hash, json.dumps(result), ex=3600)
    return result
