# syntax=docker/dockerfile:1

FROM  --platform=linux/amd64  python:3.11
#debian:bullseye-slim

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN  dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb

RUN apt-get update \
  && apt-get install -y dotnet-runtime-8.0

COPY requirements.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt


COPY alpharaw alpharaw
COPY requirements.txt requirements.txt
COPY extra_requirements extra_requirements
COPY setup.py setup.py
COPY MANIFEST.in MANIFEST.in
COPY README.md README.md

RUN pip install -e "."

RUN pip install pythonnet

ENV PYTHONNET_RUNTIME=coreclr

# build:
# docker build --progress=plain -t alpharaw .

# run bash:
# DATA_FOLDER=.
# docker run -v $DATA_FOLDER:/app/data/ -v ./alpharaw:/app/alpharaw -it alpharaw bash

# run command:
# alpharaw parse --raw /app/data/my.raw