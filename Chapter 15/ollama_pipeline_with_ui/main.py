# main.py
from agents.summarization_schema_agent import summarization_schema_agent
from tasks.aggregator import aggregate_summarized_data
from tasks.sql_generator import generate_sql
from tasks.grader import grade_sql, grade_summary
import logging

logging.basicConfig(level=logging.INFO)

def run_query(query):
    logging.info(f"Processing Query: {query}")
    
    schema_results = summarization_schema_agent.invoke({"input": query})
    aggregated_result = aggregate_summarized_data(query)

    sql_query = generate_sql(
        intent="Retrieve data",
        table="customers",
        columns=["id", "name", "age", "city"],
        conditions="product = 'shoes' AND purchase_date >= date('now', '-1 month')"
    )

    sql_grade = grade_sql(sql_query)
    summary_grade = grade_summary(aggregated_result["final_summary"])

    final_output = {
        "aggregated_result": aggregated_result,
        "sql_query": sql_query,
        "sql_grade": sql_grade,
        "summary_grade": summary_grade
    }
    
    logging.info("Pipeline Execution Completed Successfully")
    return final_output
