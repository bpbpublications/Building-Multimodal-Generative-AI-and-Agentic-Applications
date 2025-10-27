# tasks/sql_generator.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from core.llm import get_llm

llm = get_llm()

sql_prompt_template = PromptTemplate(
    input_variables=["intent", "table", "columns", "conditions"],
    template="""
    Let's break down the query into logical steps:

    1. Identify the intent of the query: {intent}
    2. Match the relevant tables: {table}
    3. Identify the relevant columns: {columns}
    4. Define conditions to filter the data: {conditions}
    5. Construct the SQL query using the identified components.

    SQL Query:
    SELECT {columns} FROM {table} WHERE {conditions};
    """
)

sql_chain = LLMChain(prompt=sql_prompt_template, llm=llm)

def generate_sql(intent, table, columns, conditions):
    return sql_chain.run(intent=intent, table=table, columns=", ".join(columns), conditions=conditions)

