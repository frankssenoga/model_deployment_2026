FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Render port
EXPOSE 10000

# Start Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
