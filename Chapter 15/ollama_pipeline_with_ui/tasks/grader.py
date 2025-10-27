# Placeholder for grader function# tasks/grader.py

from tasks.utils import ollama_completion

def grade_sql(sql_query):
    return ollama_completion(f"Grade this SQL query for correctness, relevance, and efficiency:\n{sql_query}")

def grade_summary(summary_text):
    return ollama_completion(f"Grade this summary for accuracy, comprehensiveness, and clarity:\n{summary_text}")
