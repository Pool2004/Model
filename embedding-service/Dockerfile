FROM python:3.10-slim-buster

WORKDIR /app

# Instalar dependencias del sistema necesarias para sentence-transformers o torch si es necesario
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo previamente para integrarlo en la imagen (opcional, recomendado para evitar descargas en runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
