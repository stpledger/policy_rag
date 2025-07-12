# Use a minimal Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Streamlit should not try to open a browser
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port expected by Cloud Run or App Engine Flex
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
