import mlflow
import os

# Set tracking to local 'mlruns' folder (optional but explicit)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("Default")

with mlflow.start_run() as run:
    print(f"✅ Started run: {run.info.run_id}")
    
    # Log simple param and metric
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 0.42)

    # Create and log a dummy artifact
    with open("dummy.txt", "w") as f:
        f.write("Hello from MLflow!")

    mlflow.log_artifact("dummy.txt")

    print("✅ Finished run. Check MLflow UI.")
