FROM python:3.8
#It tells docker to create an imagine that will be inherited from an image named: 3.8-slim-buster
#This command is telling the docker service to use the base image as python:3.8-slim-buster. This is an official Python image.  
#It has all of the required packages that we need to run a Python application.
WORKDIR /src
#We are setting a working directory. It will then navigate to the src folder. It is essentially creating a working directory. From now on, we can pass in the relative paths based on the src directory.
COPY requirements.txt /src/requirements.txt
ARG DEBIAN_FRONTEND=noninteractive
# Necessary operations

# RUN apt-get update
# RUN apt-get -y install python3
# RUN apt-get -y install python3-pip
# RUN python3 --version
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT [ "python" ]
CMD ["app.py" ]