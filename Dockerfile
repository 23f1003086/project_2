FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl wget ca-certificates git \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxrandr2 libxss1 libasound2 \
    libx11-6 libx11-xcb1 libxcb1 libxkbcommon0 libgbm1 \
    fonts-liberation libfontconfig1 tzdata libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY . /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

RUN playwright install --with-deps chromium

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
