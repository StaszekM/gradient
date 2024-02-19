FROM nvidia/cuda:12.0.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 
ARG UNAME
ARG USERID
ARG GID


RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $USERID -g $GID -o -s /bin/bash $UNAME


RUN apt-get update && \
    apt-get install -y software-properties-common  && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.8 python3-pip python3.8-dev python3.8-distutils && \
    rm -rf /var/lib/apt/lists/* && \
    python3.8 -m pip install --upgrade pip

RUN mkdir /app && chown ${USERID} /app 
COPY requirements.txt /app

USER $USERID

EXPOSE 9000

CMD ["tail" , "-f", "/dev/null"]