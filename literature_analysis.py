# pip install streamlit nltk pymorphy2 wordcloud gensim matplotlib
# streamlit run literature_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import re
from gensim import corpora
from gensim.models import LdaModel
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec



# Загрузка необходимых ресурсов NLTK
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Функция для очистки и предобработки текста
def pre_process_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = re.sub(r'[^\w\s]', '', text)

    # Токенизация
    words = word_tokenize(text, language='russian')  # Указываем русский язык

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    words = [word for word in words if word not in stop_words]

    # Лемматизация с использованием pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    words = [morph.parse(word)[0].normal_form for word in words]  # Получение нормальной формы слова

    return words

# Функция для тематического моделирования с использованием LDA
def lda_model(words, num_topics, passes, num_words):
    # Создание словаря и корпуса для LDA
    dictionary = corpora.Dictionary([words])
    # dictionary.filter_extremes(no_below=1, no_above=0.5)
    corpus = [dictionary.doc2bow(words)]
    
    # Обучение модели LDA
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    
    # Извлечение тем
    topics = lda.print_topics(num_words=num_words)
    topics_format = []
    
    for topic in topics:
        topic_id, topic_words = topic
        topic_id = f'Topic {int(topic_id)}'
        topics_format.append((topic_id, topic_words))
    
    return topics_format

# Функция для сентимент-анализа
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores


def word2vec_topic_modeling(text_data, num_topics=5, num_words=5):
    # Предобработка текста
    processed_data = [pre_process_text(text) for text in text_data]

    # Обучение модели Word2Vec
    model = Word2Vec(sentences=processed_data, vector_size=100,
                     window=5, min_count=1, workers=4)

    # Получение векторов для всех слов в модели
    word_vectors = model.wv.vectors  # Получаем векторы слов

    # Преобразование векторов в float64
    # Преобразуем векторы в массив NumPy с типом float64
    word_vectors = np.array(word_vectors, dtype=np.float64)

    # Кластеризация K-Means
    kmeans = KMeans(n_clusters=num_topics)
    kmeans.fit(word_vectors)

    # Получение ключевых слов для каждой темы
    topics = {f'Topic {i}': [] for i in range(num_topics)}

    for word in model.wv.index_to_key:
        # Получаем номер кластера для текущего слова
        cluster_num = kmeans.predict([model.wv[word]])[0]
        topics[f'Topic {cluster_num}'].append(word)

    # Вывод ключевых слов по темам
    result = {}
    for topic, words in topics.items():
        # Берем топ num_words слов
        top_words = words[:num_words]
        result[topic] = ' '.join(top_words)

    return result 

# Основной интерфейс Streamlit
st.title("Анализ классической литературы")


uploaded_file = st.file_uploader("Выберите текстовый файл", type=["txt"])
if uploaded_file is not None:
    string_data = uploaded_file.read().decode("cp1251")
    # st.write(string_data)

    # Предобработка текста
    words = pre_process_text(string_data)

    # Частотный анализ
    freq = nltk.FreqDist(words)

    # Отображение 10 наиболее часто употребляемых слов
    st.write("10 наиболее часто употребляемых слов:")
    st.bar_chart(dict(freq.most_common(10)))


    # Облако слов
    st.subheader("Облако слов")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off') 
    st.pyplot(plt)

    # Настройка гиперпараметров для LDA
    st.subheader("Настройка гиперпараметров модели:")
    num_topics = st.slider("Количество тем", min_value=2, max_value=10, value=4)  # Значение по умолчанию 2
    passes = st.slider("Количество проходов", min_value=1, max_value=20, value=10)  # Значение по умолчанию 10
    num_words = st.slider("Количество слов на тему", min_value=5, max_value=15, value=5)  # Значение по умолчанию 5

    # Тематическое моделирование lda
    st.subheader("Темы в тексте lda:")
    topics = lda_model(words, num_topics, passes, num_words)
    
    # Преобразование тем в DataFrame для отображения
    topics_df = pd.DataFrame(topics, columns=["Topic", "Words"])
    
    # Отображение темы в виде таблицы
    st.table(topics_df)


    # Тематическое моделирование word2vec
    st.subheader("Темы в тексте word2vec:")
    topics = word2vec_topic_modeling([string_data], num_topics, num_words)
    
    # Преобразование тем в DataFrame для отображения
    topics_df = pd.DataFrame(list(topics.items()), columns=["Topic", "Words"])
    
    # Отображение темы в виде таблицы
    st.table(topics_df)

  # Сентимент-анализ
    st.subheader("Сентимент-анализ текста")
    sentiment_scores = sentiment_analysis(string_data)
    
    st.write("""Вероятность сентиментов:
             
    neg — вероятность, что текст негативный.
    neu — вероятность, что текст нейтральный.
    pos — вероятность, что текст позитивный.
""")
    st.json(sentiment_scores)  # Показать результаты в формате JSON

    st.subheader("Анализ лексического разнообразия")
    # Анализ лексического разнообразия
    lex_diversity = len(set(words)) / len(words)
    st.write(f"Лексическое разнообразие: {lex_diversity:.2f}")

    # Средняя длина предложений
    sentences = nltk.sent_tokenize(string_data, language='russian')  # Указываем русский язык
    avg_sentence_length = np.mean([len(s.split()) for s in sentences])
    st.write(f"Средняя длина предложений: {avg_sentence_length:.2f} слов")