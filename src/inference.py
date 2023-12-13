from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-east1",
    api_endpoint: str = "us-east1-aiplatform.googleapis.com",
):
    """Make a prediction to a deployed custom trained model
    Args:
        project (str): Project ID
        endpoint_id (str): Endpoint ID
        instances (Union[Dict, List[Dict]]): Dictionary containing instances to predict
        location (str, optional): Location. Defaults to "us-east1".
        api_endpoint (str, optional): API Endpoint. Defaults to "us-east1-aiplatform.googleapis.com".
    """
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


predict_custom_trained_model(
    project="851966923939",
    endpoint_id="8107420511220269056",
    location="us-east1",
    instances= {
            "cc_freq": -0.49281,
            "city": -0.979,
            "job": -0.4493,
            "age": -0.73624,
            "gender_M": 0,
            "merchant": 2.217,
            "category": 1.507,
            "distance_km": 1.77,
            "month": -0.974,
            "day": -0.816,
            "hour": -0.123,
            "hours_diff_bet_trans": -0.71278,
            "amt": -0.44261
        }
)