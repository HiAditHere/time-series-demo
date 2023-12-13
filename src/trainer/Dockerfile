FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://credit-card-fraud-detection-group5/model
ENV BUCKET_NAME=credit-card-fraud-detection-group5

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs python-dotenv


ENTRYPOINT ["python", "-m", "trainer.train"]