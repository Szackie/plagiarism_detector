FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN python -m venv venv
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PD_env World

CMD ["venv/bin/gunicorn", "--bind", "0.0.0.0:8000", "wsgi:app"]