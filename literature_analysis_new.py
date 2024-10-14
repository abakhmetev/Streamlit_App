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
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# -- Set page config
apptitle = 'Analysis of classical literature'
st.set_page_config(page_title=apptitle, page_icon=":material/view_kanban:")

# Функция для очистки и предобработки текста
def pre_process_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации

    # Токенизация
    words = word_tokenize(text, language='russian')

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    words = [word for word in words if word not in stop_words]

    # Лемматизация с использованием pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    words = [morph.parse(word)[0].normal_form for word in words]  # Получение нормальной формы

    return words

# Функция для тематического моделирования с использованием LDA
def lda_model(words, num_topics, passes, num_words):
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
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

# Функция для тематического моделирования с использованием Word2Vec
def word2vec_topic_modeling(text_data, num_topics=5, num_words=5):
    processed_data = [pre_process_text(text) for text in text_data]
    model = Word2Vec(sentences=processed_data, vector_size=100, window=5, min_count=1, workers=4)

    word_vectors = model.wv.vectors
    word_vectors = np.array(word_vectors, dtype=np.float64)

    kmeans = KMeans(n_clusters=num_topics)
    kmeans.fit(word_vectors)

    topics = {f'Topic {i}': [] for i in range(num_topics)}

    for word in model.wv.index_to_key:
        cluster_num = kmeans.predict([model.wv[word]])[0]
        topics[f'Topic {cluster_num}'].append(word)

    result = {}
    for topic, words in topics.items():
        top_words = words[:num_words]
        result[topic] = ' '.join(top_words)

    return result 

# Основной интерфейс Streamlit
st.title("Анализ классической литературы")


# Перемещение загрузки файла и гиперпараметров на боковую панель
st.sidebar.header("Настройки")
uploaded_file = st.sidebar.file_uploader("Выберите текстовый файл", type=["txt"])

# Выбор модели
model_selection = st.sidebar.selectbox("Выберите модель:", ["Word2Vec", "LDA"])

# Гиперпараметры для LDA и Word2Vec
st.sidebar.subheader("Настройка гиперпараметров:")
num_topics = st.sidebar.slider("Количество тем", min_value=2, max_value=10, value=4)
passes = st.sidebar.slider("Количество проходов", min_value=1, max_value=20, value=10)
num_words = st.sidebar.slider("Количество слов на тему", min_value=5, max_value=15, value=5)

if uploaded_file is not None:
    string_data = uploaded_file.read().decode("cp1251")

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

    # Выбор модели для тематического моделирования
    if model_selection == "LDA":
        st.subheader("Темы в тексте LDA:")
        topics = lda_model(words, num_topics, passes, num_words)
        topics_df = pd.DataFrame(topics, columns=["Topic", "Words"])
        st.table(topics_df)

    elif model_selection == "Word2Vec":
        st.subheader("Темы в тексте Word2Vec:")
        topics = word2vec_topic_modeling([string_data], num_topics, num_words)
        topics_df = pd.DataFrame(list(topics.items()), columns=["Topic", "Words"])
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

    # Анализ лексического разнообразия
    st.subheader("Анализ лексического разнообразия")
    lex_diversity = len(set(words)) / len(words)
    st.write(f"Лексическое разнообразие: {lex_diversity:.2f}")

    # Средняя длина предложений
    sentences = nltk.sent_tokenize(string_data, language='russian')  # Указываем русский язык
    avg_sentence_length = np.mean([len(s.split()) for s in sentences])
    st.write(f"Средняя длина предложений: {avg_sentence_length:.2f} слов")
