# Используем slim-образ Python
FROM python:3.11-slim

# Неинтерактивная установка
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Рабочая директория
WORKDIR /app

# Установка зависимостей системы
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    wget \
    git \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы проекта
COPY . /app

# Устанавливаем Python зависимости
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Создаем директории для данных и логов
RUN mkdir -p /app/data/logs /app/data/temp /app/charts

# Экспорт переменных окружения по умолчанию (можно переопределить в Railway)
ENV DATA_DIR=/app/data
ENV DATABASE_PATH=/app/data/trading_bot.db

# Команда запуска бота
CMD ["python", "bot.py"]