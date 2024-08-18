FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN python -m venv venv
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Create a directory for NLTK data and set NLTK_DATA environment variable
RUN mkdir -p /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

# Download wordnet to the specified directory
RUN . venv/bin/activate && python venv/lib/python3.9/site-packages/nltk/downloader.py -d /app/nltk_data wordnet

EXPOSE 8000

ENV PD_env World

CMD ["venv/bin/gunicorn", "--bind", "0.0.0.0:8000", "wsgi:app"]