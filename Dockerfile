ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
COPY main.py .
RUN apt update \
    && apt install dnsutils -y