def build_translation_dict():
    result = {}
    QWERTY = "qwertyuiop[]asdfghjkl;'zxcvbnm`,."
    JCUKEN = "йцукенгшщзхъфывапролджэячсмитьёбю"
    for (latin, cyrillic) in zip(QWERTY, JCUKEN):
        result[ord(latin)] = cyrillic
        # Handle letter case pairs
        if latin.islower():
            result[ord(latin.upper())] = cyrillic.upper()
    return result

def AltShift(query):
    return str.translate(query, build_translation_dict())
