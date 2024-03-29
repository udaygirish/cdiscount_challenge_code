# Docker File for Deploying the eval system - Main purpose

FROM ubuntu:focal
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install python3-pip
RUN apt-get install -y ffmpeg libsm6  libxext6

# For additional requirements please copy optional_requirements too 
# For base apps the step mentioned in above comment is not necessary
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 5001
EXPOSE 8501

RUN ["chmod","+x", "./commands.sh"]
ENTRYPOINT [ "./commands.sh" ]
