FROM python:3.8-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv and cython
RUN pip install pipenv cython

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock /app/

# Install dependencies
RUN pipenv install --deploy --ignore-pipfile

# Copy the rest of the application
COPY . /app

# Command to run the application
CMD ["pipenv", "run", "gunicorn", "-w", "4", "-k", "gevent", "-b", "0.0.0.0:8000", "app:app"]
