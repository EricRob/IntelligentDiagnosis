FROM tensorflow/tensorflow:1.9.0-gpu-py3
COPY requirements.txt /opt/out/
WORKDIR /opt/out/
RUN pip3 install -r requirements.txt