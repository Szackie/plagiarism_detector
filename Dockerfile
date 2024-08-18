FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN python -m venv /app/venv

RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PD_env World

CMD ["/app/venv/bin/python", "--bind", "0.0.0.0:8000", "wsgi:app"]