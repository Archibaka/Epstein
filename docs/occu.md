# Модуль Occu.py

## Описание
Система индексации PDF-файлов и поиска:
1. Конвертирует PDF в текстовые фрагменты
2. Создает векторные эмбеддинги
3. Сохраняет в ChromaDB
4. Осуществляет семантический поиск

## Зависимости
```bash
pip install haystack-ai chroma-haystack pdfplumber
```

Модули <a href="https://github.com/Archibaka/Epstein/tree/main/docs/conv.md">conv</a>, <a href="https://github.com/Archibaka/Epstein/tree/main/docs/lilim.md">Lilim</a>

## Требования
Загруженные в 
```python
path = "./files/GOOD"
```
PDF-файлы с имеющимся текстом (для отсканированных PDF используйте <a href="https://github.com/Archibaka/Epstein/tree/main/docs/toNormal.md">toNormal.py</a>)

Ser.py, запущенный на 

```python
hostip = "http://localhost:8080""
```

Загруженная в 
```python
path = "./models/Jeffry/qwen3"
```
LLM, совместимая с Hugging Face transformers

### Использование
```python
from Occu import ind, ret
```
# Индексация PDF из ./files/list
```python
ind()
```
или 

### Запуск

```bash
python occu.py
```

Сформированная векторная база данных будет храниться в 
```python
persist_path="./chroma_db"
```

# Поиск в базе
```python
results = ret("Ваш запрос")
```
Особенности
Автоматическая обработка директории

Интеграция с Ser.py для эмбеддингов

Поддержка многопоточности (test.py)
