FROM tensorflow/tensorflow:2.4.1

RUN mkdir app
#COPY * app/
RUN ls app

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt install -y pkg-config libopencv-dev libjpeg-dev libpng-dev libtiff-dev git screen nano curl python3-dev python3-opencv python3-pycurl python3-numpy

COPY src app/src
COPY data app/data
COPY label-maps app/label-maps
COPY models app/models
COPY object_detection app/object_detection
COPY requirements.txt app/requirements.txt
COPY requirements-nodeps.txt app/requirements-nodeps.txt
COPY script.sh app/script.sh

WORKDIR /app
ENV PYTHONPATH /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-deps -r requirements-nodeps.txt
RUN ls

CMD bash script.sh
EXPOSE 5000
