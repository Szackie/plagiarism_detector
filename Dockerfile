FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN python -m venv venv
RUN . venv/bin/activate
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
ENV PD_env
CMD ["python", "app2.py"]