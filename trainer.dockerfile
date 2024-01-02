# Base image
FROM --platform=linux/amd64 python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
# COPY setup.py setup.py
COPY mlops_template/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

