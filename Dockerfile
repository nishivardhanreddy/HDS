FROM python:3.11-slim

# Avoid unnecessary Python cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependencies first (faster builds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
