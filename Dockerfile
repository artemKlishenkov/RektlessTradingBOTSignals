# Используем официальный образ Python 3.12 slim
FROM python:3.12-slim

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем базовые build tools и обновляем pip, setuptools, wheel
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && apt-get remove -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Копируем весь проект
COPY . .


# Команда запуска
CMD ["python", "bot.py"]