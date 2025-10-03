import mlflow
import os
from packaging.version import Version
import logging

@mlflow.trace(span_type="func", attributes={"name": "add_1"})
def add_1(x):
    return x + 1

@mlflow.trace(span_type="func")
def mybot_ask(question, response):
    add_1(123)
    return response

if __name__ == "__main__":
    # Uncomment the following line to ensure the correct version of mlflow is used
    # assert Version(mlflow.__version__) >= Version("3.0.0"), (
    #     "Not using mlflow 3.0.0. Using " + str(mlflow.__version__)
    # )
    
    # Set the endpoint of the OpenTelemetry Collector
    # os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:8080/v1/traces"
    
    logger = logging.getLogger("mlflow")

    # Set log level to debugging
    # logger.setLevel(logging.DEBUG)

    # mlflow.set_tracking_uri("http://localhost:8910")
    mlflow.set_experiment("Gemini with OpenAI3")
    
    with mlflow.start_run() as run:
        print("run id " + run.info.run_id)
        print(mybot_ask("What is the capital of France?", "Paris"))
        print(mybot_ask("What is 2 + 2?", "4"))
        print(mybot_ask("Who wrote '1984'?", "George Orwell"))
        print(mybot_ask("What is the largest planet in our solar system?", "Jupiter"))
        print(mybot_ask("What is the boiling point of water in Celsius?", "100"))
        print(mybot_ask("Who painted the Mona Lisa?", "Leonardo da Vinci"))
        print(mybot_ask("What is the chemical symbol for gold?", "Au"))
        print(mybot_ask("What is the capital of Japan?", "Tokyo"))
        print(mybot_ask("Who discovered penicillin?", "Alexander Fleming"))
        print(mybot_ask("What is the square root of 64?", "8"))
        print(mybot_ask("What is the speed of light in vacuum (m/s)?", "299,792,458"))


