FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

COPY . .

RUN echo "Downloading model with RUN_ID=${RUN_ID}"

CMD ["python", "train.py"]