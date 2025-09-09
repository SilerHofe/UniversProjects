import nltk
from lab1 import sentences
import lab2
from natasha import (
    Doc,
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser
)

# Инициализация компонентов Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

# Возьмем первое предложение из нашего датасета для примера
example_sentence = sentences[0]
print(f"6. СИНТАКСИЧЕСКИЙ РАЗБОР ПРЕДЛОЖЕНИЯ:")
print(f"Предложение: \"{example_sentence}\"")
print("\n" + "-"*50 + "\n")

# Создаем объект Doc для предложения
doc = Doc(example_sentence)
# Сегментация (в данном случае - разбивка на токены)
doc.segment(segmenter)
# Морфологический разбор
doc.tag_morph(morph_tagger)
# Синтаксический разбор
doc.parse_syntax(syntax_parser)

# Визуализация дерева зависимостей
print("ДЕРЕВО ЗАВИСИМОСТЕЙ (Natasha):")
# Обходим каждый токен и выводим информацию о его синтаксической голове
for token in doc.tokens:
    # token.head_id - id головного токена, token.rel - тип синтаксической связи
    print(f"Токен: '{token.text}' (ID: {token.id}) | Глава: {token.head_id} | Связь: {token.rel}")

# Для наглядности можно представить дерево в виде списка кортежей (токен, глава, связь)
syntax_tree = [(token.text, token.head_id, token.rel) for token in doc.tokens]
print(f"\nСписок зависимостей: {syntax_tree}")

# Попытка использовать встроенный парсер NLTK (для демонстрации его неэффективности)
print("\n" + "="*50)
print("ПОПЫТКА СИНТАКСИЧЕСКОГО РАЗБОРА СРЕДСТВАМИ NLTK (RegexpParser):")
# Создаем простейшую контекстно-свободную грамматику для примера
grammar = "NP: {<ADJ>*<NOUN>+}"
try:
    # Для использования этого парсера нужны предварительно размеченные по частям речи токены.
    # Т.к. NLTK плохо справляется с русским, используем результаты Natasha.
    # Конвертируем разбор Natasha в формат, понятный NLTK (список кортежей (word, pos))
    tagged_tokens = [(token.text, token.pos) for token in doc.tokens]
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged_tokens)
    print("Результат разбора по шаблону 'Прилагательное* + Существительное+' (NP - именная группа):")
    print(result)
except Exception as e:
    print(f"NLTK RegexpParser не сработал корректно: {e}")

# ВЫВОД: Natasha предоставляет готовый и качественный синтаксический разбор для русского языка,
# в то время как стандартные средства NLTK требуют доработки и дают лишь базовые результаты.