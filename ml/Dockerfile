FROM python:3.8.5-slim-buster as base
FROM base as builder
WORKDIR /usr/src/app
RUN apt-get update -y \ 
    && DEBIAN_FRONTEND=noninteractive apt -y install gcc

RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

FROM base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

RUN useradd -ms /bin/bash app
RUN chown app /usr/share/
ENV HOME=/home/app
ENV APP_HOME=/home/app/ml

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libtinfo5 \
    && apt-get --fix-broken install\
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/*

USER app
WORKDIR $APP_HOME

ENV PATH=$PATH:$HOME/.local/bin
COPY --from=builder /usr/src/app/wheels /wheels

RUN pip install --upgrade --no-cache /wheels/* \
     && rm -rf /tmp/* 

COPY . $APP_HOME
