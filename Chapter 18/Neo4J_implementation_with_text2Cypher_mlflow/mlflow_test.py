import mlflow

with mlflow.start_run():
    print("✅ Test run started")
    mlflow.log_param("test_param", "value")
    mlflow.log_metric("test_metric", 42)
    print("✅ Test run completed")
