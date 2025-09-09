from pymorphy3 import MorphAnalyzer
from lab1 import filtered_words

# Инициализация морфологического анализатора
morph = MorphAnalyzer(lang='ru')

# Возьмем для разбора первые 15 слов из очищенного списка
words_to_analyze = filtered_words[:15]

print("\n 5. МОРФОЛОГИЧЕСКИЙ РАЗБОР (с использованием pymorphy3):")
for word in words_to_analyze:
    parse = morph.parse(word)[0]  # Берем самый вероятный разбор
    # parse.tag - объект, содержащий всю морфологическую информацию
    print(f"Слово: '{word}'")
    print(f"  Нормальная форма (лемма): {parse.normal_form}")
    print(f"  Часть речи: {parse.tag.POS}")
    print(f"  Падеж: {parse.tag.case}")
    print(f"  Число: {parse.tag.number}")
    print(f"  Время (для глаголов): {parse.tag.tense}")
    print(f"  Полный разбор: {parse.tag}")
    print()