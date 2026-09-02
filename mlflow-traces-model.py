import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

# ── Setup ──────────────────────────────────────────────────────────────────────
# For local tracking (no server needed), remove the line below.
# mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "demo-experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow version: {mlflow.__version__}")


# ── Simple PyfuncModel ─────────────────────────────────────────────────────────
class SimpleModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return (model_input * 2.0).values


# ── Main run ───────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="experiment-with-model") as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # -- Parameters & metrics
    mlflow.log_params({
        "n_samples": 100,
        "n_features": 4,
        "learning_rate": 0.01,
    })
    mlflow.log_metrics({
        "accuracy": 0.92,
        "loss": 0.08,
        "f1_score": 0.89,
    })

    # -- Sample data
    rng = np.random.default_rng(42)
    cols = [f"feature_{i}" for i in range(4)]
    X = pd.DataFrame(rng.random((100, 4)), columns=cols)
    y = rng.random(100)

    # -- Log a dataset artifact (optional but useful)
    X.to_csv("dataset.csv", index=False)
    mlflow.log_artifact("dataset.csv")

    # -- Train/fit and log model
    model = SimpleModel()
    sample_output = model.predict(None, X[:5])
    signature = infer_signature(X, sample_output)

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        signature=signature,
        input_example=X[:5],
    )

    print(f"Model logged at: {model_info.model_uri}")

    # -- Add traces manually (works on MLflow >= 2.x without tracing extras)
    with mlflow.start_span("preprocess") as span:
        span.set_inputs({"shape": list(X.shape)})
        X_norm = X / X.max()
        span.set_outputs({"shape": list(X_norm.shape), "max_val": float(X_norm.max().max())})
        span.set_attribute("step", "normalization")

    with mlflow.start_span("inference") as span:
        span.set_inputs({"n_rows": len(X_norm)})
        preds = model.predict(None, X_norm)
        span.set_outputs({"n_predictions": len(preds), "sample": preds[:3].tolist()})
        span.set_attribute("step", "model_predict")

print("\n✅ Done!")
print(f"  Run ID  : {run_id}")
print(f"  Model   : runs:/{run_id}/model")
print(f"\nTo load model:")
print(f'  loaded = mlflow.pyfunc.load_model("runs:/{run_id}/model")')
print(f'  loaded.predict(X)')
