FROM python:3.11
COPY . /procting
WORKDIR /procting
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y git pkg-config libopencv-dev
RUN pip install -r requirements.txt
CMD python proctor.py