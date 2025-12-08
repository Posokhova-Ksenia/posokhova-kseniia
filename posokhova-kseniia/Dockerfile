# Базовый образ Python
FROM python:3.10-slim


# Устанавливаем системные зависимости для LightGBM и других пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .


# Установка всех необходимых пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда запуска вашего решения
CMD ["python", "main.py"]
