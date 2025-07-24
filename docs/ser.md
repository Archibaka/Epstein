# Модуль Ser.py

## Описание
Локальный сервер эмбеддингов на базе ru-en-RoSBERTa:
- Запускает Flask-сервер
- Принимает тексты, возвращает векторы
- Интегрируется с Haystack

## Зависимости
```bash
pip install flask sentence-transformers
```

## Требования 
Загруженная в 
```python
model = SentenceTransformer("./models/embodel/RoSBERTa")
```
embedding model, совместимая с Hugging Face Sentence Transformers

## Конфигурация
По умолчанию:

Хост: localhost

Порт: 8080

Изменение в коде:

```python
app.run(host="ваш_хост", port="ваш_порт")
```

<strong> При изменении обязательно поменять и в <a href="https://github.com/Archibaka/Epstein/tree/main/docs/occu.md">occu.py</a></strong>
## API
POST / - Основной эндпоинт
Форматы запросов:

TEI-совместимый формат (для HuggingFace):

```json
{
  "inputs": ["текст1", "текст2"]
}
```
Упрощенный формат:

json
["текст1", "текст2"]
Ответ:

```json
[
  [0.12, -0.45, ..., 0.78], // Вектор для текста1
  [0.34, 0.56, ..., -0.12]   // Вектор для текста2
]
```
Примеры:

Запрос с одним текстом:

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"inputs": "один текст"}'
```
Запрос с массивом текстов:

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '["первый текст", "второй текст"]'
```
Обработка ошибок

Возвращает HTTP 400 при:

Отсутствии текстовых данных

Неподдерживаемом формате запроса

```json
{
  "error": "Invalid input format"
}
```
