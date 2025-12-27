# Базовый образ Python
FROM python:3.10-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .

# Системные зависимости для LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Установка всех необходимых пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда запуска вашего решения
CMD ["python", "main.py"]
