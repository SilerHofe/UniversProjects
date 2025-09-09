# Импорт необходимых библиотек
import nltk
from multipart import file_path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Необходимо скачать ресурсы NLTK (выполняется один раз)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
file_path = "text.txt"

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"Файл {file_path} не найден!")
    exit()
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    exit()

# 1. Токенизация предложений
sentences = sent_tokenize(text, language="russian")
print("1. ТОКЕНИЗАЦИЯ ПРЕДЛОЖЕНИЙ:")
for i, sent in enumerate(sentences, 1):
    print(f"Предложение {i}: {sent}")
print("\n" + "-"*50 + "\n")

# 2. Токенизация слов для всего текста
tokens = word_tokenize(text, language="russian")
print(f"2. ТОКЕНИЗАЦИЯ СЛОВ. Всего слов: {len(tokens)}")
print(f"Первые 20 токенов: {tokens[:20]}")
print("\n" + "-"*50 + "\n")

# 3. Нормализация: приведение к нижнему регистру и удаление не-буквенных токенов (знаки препинания)
words = [word.lower() for word in tokens if word.isalpha()]
print(f"3. НОРМАЛИЗАЦИЯ. Только слова в нижнем регистре: {len(words)}")
print(f"Пример: {words[:15]}")
print("\n" + "-"*50 + "\n")

# 4. Подключение стоп-слов и их удаление
russian_stopwords = stopwords.words("russian")
additional_stopwords = ['это', 'вот', 'как', 'так', 'уже', 'еще', 'тот', 'свой']
russian_stopwords.extend(additional_stopwords)

filtered_words = [word for word in words if word not in russian_stopwords]

print(f"4. УДАЛЕНИЕ СТОП-СЛОВ.")
print(f"Размер стоп-листа: {len(russian_stopwords)}")
print(f"Примеры стоп-слов: {russian_stopwords[:10]}...")
print(f"Слов после удаления стоп-слов: {len(filtered_words)}")
print(f"Пример результата: {filtered_words[:15]}")