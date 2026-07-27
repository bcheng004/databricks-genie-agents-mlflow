"""Preset Guidelines judges for the Genie MLflow app (from notebook 02).

These seed the Guidelines section on the Evaluate page and are also available
as quick-add options on the "Create Guideline Judge" page. Each value is a list
of guideline strings passed to ``mlflow.genai.scorers.Guidelines(guidelines=...)``.
"""

GUIDELINE_JUDGES: dict[str, list[str]] = {
    "genie_response_quality": [
        "The response must directly address the user's data question "
        "rather than giving a vague or generic reply.",
        "If SQL was generated, the response must include a data-driven "
        "answer, not just echo the SQL query back.",
        "The response must not say 'I cannot answer' when the question "
        "is about data that should be available in the tables.",
    ],
    "genie_sql_quality": [
        "If SQL is present, it must use appropriate aggregation "
        "functions (SUM, COUNT, AVG) matching the user's intent.",
        "The SQL must include appropriate WHERE clauses to filter "
        "data as the user requested.",
        "The SQL must not use SELECT * on large tables without a "
        "LIMIT or specific filter.",
    ],
}
