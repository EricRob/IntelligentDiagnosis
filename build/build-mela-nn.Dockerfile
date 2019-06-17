# This image is available in docker hub as ericrob/intelligentdiagnosis_mela-nn:latest
FROM tensorflow/tensorflow:1.9.0-gpu-py3
COPY requirements.txt /opt/out/
WORKDIR /opt/out/
RUN pip3 install -r requirements.txt