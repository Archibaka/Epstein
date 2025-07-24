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
Параметры:

model_path: Путь к модели

ass: Включать ли при загрузке модели пропт для ассистента в RAG системе

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
#  <a href="https://huggingface.co/docs/transformers/en/main_classes/text_generation"> Параметры генерации </a>

max_new_tokens: Макс. количество новых токенов (0-32,768)

temperature: Креативность (0-1)

top_p: Фильтрация ядра выборки

top_k: Ограничение словаря

sample: Включить стохастичность

min_p: Минимальная вероятность

exponential_decay_length_penalty: Экспоненциальный штраф на количество токенов (С какого токена начинается, величина степени)

think: Наличие/отсутвие генерации токенов thinking, улучшающих качество вывода за счёт увеличения времени генерации 

cache_implementation: Способы хранения контекста
