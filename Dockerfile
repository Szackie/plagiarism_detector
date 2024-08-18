FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN python -m venv venv
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

RUN venv/bin/python -m nltk.downloader -d /app/venv/nltk_data wordnet punkt_tab

EXPOSE 8000

ENV PD_env World

CMD ["venv/bin/gunicorn", "--bind", "0.0.0.0:8000", "wsgi:app"]