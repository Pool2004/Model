FROM python:3.10-slim-buster

WORKDIR /app

# Instalar dependencias para compilar agregados de chromadb si hace falta
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
