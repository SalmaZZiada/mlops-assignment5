import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Mlops-Assignment5")  
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

if accuracy < 0.85:
    raise Exception("Model accuracy below threshold")

print("Model passed threshold")