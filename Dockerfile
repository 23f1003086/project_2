# Use a slim Python image
FROM python:3.10-slim


# Set environment
ENV PYTHONUNBUFFERED=1 \
POETRY_VIRTUALENVS_CREATE=false \
PIP_NO_CACHE_DIR=1


# Install system deps required by Playwright and common libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential curl wget ca-certificates git \
libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxrandr2 libxss1 libasound2 libx11-6 libx11-xcb1 libxcb1 libxkbcommon0 libgbm1 \
fonts-liberation libfontconfig1 tzdata \
&& rm -rf /var/lib/apt/lists/*


# Create app dir
WORKDIR /app


# Copy requirements and application
COPY requirements.txt /app/requirements.txt
COPY . /app


# Install Python deps
RUN pip install --upgrade pip setuptools wheel \
&& pip install -r /app/requirements.txt


# Install Playwright browsers (chromium). This also installs additional OS packages.
RUN playwright install --with-deps chromium


# Expose port used by Uvicorn
EXPOSE 8000


# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]