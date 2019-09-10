FROM tensorflow/tensorflow:1.9.0-gpu-py3
COPY requirements.txt /opt/out/
WORKDIR /opt/out/
ENV DISPLAY :0
RUN apt-get update && apt-get -y install python3-tk
RUN pip3 install -r requirements.txt
