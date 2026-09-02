import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "demo-experiment-with-traces"
mlflow.set_tracking_uri("http://localhost:5000")  # adjust to your tracking server
mlflow.set_experiment(EXPERIMENT_NAME)

# ── A simple model to log ──────────────────────────────────────────────────────
class SimpleModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input * 2.0


# ── Helper: a traced function ──────────────────────────────────────────────────
@mlflow.trace(name="preprocess", span_type="FUNC")
def preprocess(data: np.ndarray) -> np.ndarray:
    return data / data.max()


@mlflow.trace(name="run_inference", span_type="LLM")
def run_inference(model, data: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
    return model.predict(df)


# ── Main run ───────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="traced-run-with-model") as run:
    print(f"Run ID: {run.info.run_id}")

    # Log hyperparameters and metrics
    mlflow.log_param("n_samples", 100)
    mlflow.log_param("n_features", 4)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.08)

    # Generate sample data
    rng = np.random.default_rng(42)
    X = rng.random((100, 4))
    y = rng.random(100)

    # ── Traced calls ────────────────────────────────────────────────────────────
    # These produce spans visible in the MLflow UI under the Traces tab
    X_processed = preprocess(X)

    model_instance = SimpleModel()
    predictions = run_inference(model_instance, X_processed)

    # ── Log the model ───────────────────────────────────────────────────────────
    signature = infer_signature(
        pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]),
        predictions,
    )

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_instance,
        signature=signature,
        input_example=pd.DataFrame(X[:5], columns=[f"feature_{i}" for i in range(X.shape[1])]),
    )

    print(f"Model URI: runs:/{run.info.run_id}/model")
    print(f"View run at: {mlflow.get_tracking_uri()}/#/experiments/.../runs/{run.info.run_id}")

print("Done. Check the MLflow UI → Experiments and Traces tabs.")
