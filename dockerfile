# Use an appropriate base image, e.g., python:3.10-slim
FROM python:3.10-slim

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1
ENV HOST 0.0.0.0
ENV PORT 8000

# Set the working directory
WORKDIR /app

# Copy your application's requirements and install them
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

# Copy your application code into the container
COPY . /app/

EXPOSE $PORT

CMD python -m chainlit run app.py --host $HOST --port $PORT
