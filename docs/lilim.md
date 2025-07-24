# Модуль Lilim.py

## Описание
Обертка для работы с языковой моделью (LLM). Поддерживает:
- Локальную загрузку моделей (по умолчанию Qwen3-1.7B)
- Управление контекстом диалога
- Настройку параметров генерации

## Зависимости
```bash
pip install transformers torch
```
Модуль <a href="https://github.com/Archibaka/Epstein/tree/main/docs/conv.md">conv</a>

## Требования
Загруженная в 
```
path = "./models/Jeffry/qwen3"
```
LLM, совместимая с Hugging Face transformers

### Запуск
Отдельный запуск не требуется, но при запуске сам по себе
```
python lilim.py
```
Запустит тестовую задачу на переформулировку запроса: 
```
query = "ye;ty ujcn yf hfphf,jnre gj"
```
