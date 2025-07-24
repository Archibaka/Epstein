# Модуль conv.py

## Описание
Модуль для изменения раскладки строки с
```python
QWERTY = "qwertyuiop[]asdfghjkl;'zxcvbnm`,."
```
на 
```python
JCUKEN = "йцукенгшщзхъфывапролджэячсмитьёбю"
```

### Использование
```python
from conv import AltShift
```

```python
another_layout_str = AltShift(one_layout_str)
```

Copied from https://stackoverflow.com/questions/78010107/how-to-translate-symbols-from-latin-to-cyrillic
