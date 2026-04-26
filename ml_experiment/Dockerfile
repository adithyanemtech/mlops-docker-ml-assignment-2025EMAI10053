# Use Python base image
FROM python:3.9-slim [cite: 33]

# Set working directory
WORKDIR /app [cite: 34]

# Copy code
COPY . /app [cite: 35]

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt [cite: 37]

# Run experiment
CMD ["python", "palmer-panguin-decisionTree.py"] [cite: 38]
