# Инструкция по запуску модели

## Подключение к серверу
```bash
ssh ii@192.168.18.130
```
При появлении запроса введите пароль:
```
18
```

## Активация окружения
В любой директории выполните:
```bash
conda activate llamacpp
```

## Переход в директорию с vllm
```bash
cd /home/ii/1C/vLLM
```

## Запуск модели
В этой директории запустите модель (например, `gpt-oss-120b`):
```bash
./start_gpt.sh
```

## Ожидание полного запуска
Дождитесь сообщения:
```
INFO: Application startup complete.
```

## Запуск бека
Откройте новый терминал, подключитесь к серверу и в директории /home/ii/1C/gpt-oss/services/mcp_server поднимите бек
```bash
cd /home/ii/1C/gpt-oss/services/mcp_server
```
```bash
uvicorn mcp_server:app --host 0.0.0.0 --port 9003
```

## Повторная активация окружения
```bash
conda activate llamacpp
```

## Запуск сервисов через docker-compose
Откройте новый терминал, подключитесь к серверу, перейдите в директорию:
```bash
cd /home/ii/1C/gpt-oss
```

Если сервисы не запущены (проверьте через `docker ps`), выполните:
```bash
docker-compose up --build
```

## Запуск приложения
В директории `/home/ii/1C/gpt-oss` выполните:
```bash
python gradio_app_gpt_oss.py
```

Ожидайте сообщения:
```
Running on local URL: http://0.0.0.0:7860
```

## Тестирование модели
Откройте в браузере:
```
http://0.0.0.0:7860
```
и проверьте работу модели.
