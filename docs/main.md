# Модуль main.py

## Описание
Графический интерфейс чат-бота:
- Интегрирует все компоненты системы
- Реализует RAG-паттерн
- Локальный веб-интерфейс

## Зависимости
```
pip install flask 
```
Модули  <a href="https://github.com/Archibaka/Epstein/tree/main/docs/lilim.md">Lilim</a>, <a href="https://github.com/Archibaka/Epstein/tree/main/docs/occu.md">Occu</a>, <a href="https://github.com/Archibaka/Epstein/tree/main/docs/conv.md">conv</a>, <a href="https://github.com/Archibaka/Epstein/tree/main/docs/ser.md">Ser</a>

Содержимое папки <a href="https://github.com/Archibaka/Epstein/tree/main/static">static</a> 

##Требования
Запущенный Ser.py (localhost:8080 <strong>если Ser.py настроен на другой, обязательно поменять в Occu.py</strong>)

Проиндексированные PDF (через Occu.py)

##Запуск
```
python main.py 
```
