# services/pdf-service/Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы требований
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем непривилегированного пользователя СНАЧАЛА
RUN useradd --create-home --shell /bin/bash app

# Копируем код приложения
COPY . .

# Создаем директории и устанавливаем правильные права доступа
RUN mkdir -p /app/logs /app/uploads /app/temp \
    && chown -R app:app /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/uploads /app/temp /app/logs

# Переключаемся на непривилегированного пользователя
USER app

# Открываем порт
EXPOSE 4300

# Команда запуска PDF сервиса
CMD ["python", "pdf_server.py"]