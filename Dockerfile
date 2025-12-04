FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install minimal runtime deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
