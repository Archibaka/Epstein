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
```python
path = "./models/Jeffry/qwen3"
```
LLM, совместимая с Hugging Face transformers

### Запуск
Отдельный запуск не требуется, но при запуске сам по себе
```bash
python lilim.py
```
Запустит тестовую задачу на переформулировку запроса: 
```python
query = "ye;ty ujcn yf hfphf,jnre gj"
```

### Использование
```python
from Lilim import Lilim
```

# Инициализация модели
```python
llm = Lilim("./models/Jeffry/qwen3")
llm.load_model()
```

# Генерация текста
```python
response = llm.generate(
    "Ваш запрос",
    max_new_tokens=512,
    temperature=0.7
)
```
# Управление историей
```python
llm.add_to_history("user", "Новый запрос")
llm.clear_history()
```
# Параметры генерации
max_new_tokens: Макс. количество новых токенов

temperature: Креативность (0-1)

top_p: Фильтрация ядра выборки

top_k: Ограничение словаря

sample: Включить стохастичность
