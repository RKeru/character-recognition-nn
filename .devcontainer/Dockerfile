FROM python:3.10.9-bullseye

# Make workspace directory
RUN mkdir -p /nn

# Copy dependencies packages
COPY ./requirements.txt /nn/requirements.txt

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y

RUN pip3 install install -r /nn/requirements.txt