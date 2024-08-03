FROM python:3.9-slim

# Set environment variables
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3


# Set environment variables for Spark
ENV SPARK_HOME=/app/spark-3.5.1-bin-hadoop3
ENV PATH="$PATH:$SPARK_HOME/bin"
ENV JAVA_HOME=/app/java-19-openjdk-amd64
ENV PATH="$PATH:$JAVA_HOME/bin"
ENV PATH="$PATH:$SPARK_HOME/sbin"



WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "dockerPredictions.py"]
