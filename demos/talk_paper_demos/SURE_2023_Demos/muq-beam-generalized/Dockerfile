FROM ubuntu:latest

COPY server.py /

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y python3-pip && \
    pip install umbridge

CMD python3 server.py
