FROM python:3.12-slim

WORKDIR /app

# Install system dependencies - use the working pattern
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl \
    build-essential \
    ffmpeg \
    libnss3 libatk1.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxrandr2 libxi6 libxss1 \
    libglib2.0-0 libgtk-3-0 \
    libasound2 libgbm1 libdrm2 libxdamage1 libxfixes3 \
    libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright explicitly and install browsers
RUN pip install playwright
RUN playwright install --with-deps chromium

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]