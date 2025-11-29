# Use the official slim Python image for reduced size and stability
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install necessary system dependencies for Playwright, PDFs, and general utilities.
# This list is comprehensive for running Chromium headless on a Debian-based image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    wget gnupg unzip curl \
    build-essential \
    ffmpeg \
    libnss3 libatk1.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxrandr2 libxi6 libxss1 \
    libglib2.0-0 libgtk-3-0 \
    libasound2 libgbm1 libdrm2 libxdamage1 libxfixes3 \
    libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 \
    # Clean up to minimize image size
    && rm -rf /var/lib/apt/lists/*

# 2. Copy and install Python dependencies from requirements.txt.
# This step is highly cacheable.
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 3. Install Playwright's browser executables (Chromium).
# This is crucial for the async_playwright context manager to launch a browser.
RUN playwright install chromium

# 4. Copy the rest of the application files (including app.py, .env, etc.)
# If you create a .dockerignore file, it will prevent unwanted files from being copied.
COPY . /app

# 5. Expose the port used by Uvicorn/FastAPI
EXPOSE 8000

# 6. Command to run the application using Uvicorn
# 'app:app' assumes your FastAPI instance is named 'app' inside the 'app.py' file.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]