FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY alembic/ ./alembic/

EXPOSE 9001

CMD ["sh", "-c", "alembic -c src/alembic.ini upgrade head && uvicorn src.app:app --host 0.0.0.0 --port 9001"]
