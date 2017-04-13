from gcr.io/tensorflow/tensorflow:latest-gpu-py3
WORKDIR /app
COPY ./project /app
RUN pip install -r requirements.txt

