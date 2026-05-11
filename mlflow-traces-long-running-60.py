import time
import logging
import mlflow


@mlflow.trace(span_type="func", attributes={"name": "add_1"})
def add_1(x):
    return x + 1


@mlflow.trace(span_type="func")
def mybot_ask(question, response):
    add_1(123)
    return response


@mlflow.trace(span_type="func", attributes={"name": "long_sleep"})
def long_running_step(seconds):
    # Sleep to simulate a long-running execution.
    time.sleep(seconds)
    return f"Slept for {seconds} seconds"


if __name__ == "__main__":
    logger = logging.getLogger("mlflow")

    # Set log level to debugging if needed.
    # logger.setLevel(logging.DEBUG)

    mlflow.set_experiment("Gemini with OpenAI3 - Long Running")

    with mlflow.start_run() as run:
        print("run id " + run.info.run_id)
        print(mybot_ask("What is the capital of France?", "Paris"))
        print("Starting long-running step (60 seconds)...")
        print(long_running_step(60))
        print(mybot_ask("What is 2 + 2?", "4"))
        print("Execution complete.")
