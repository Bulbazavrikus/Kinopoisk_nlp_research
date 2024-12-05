import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import pymorphy3
import re
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt


tqdm.pandas()    


def grade10_fix(df):

    # Удалим битые наблюдения
    df = df[df['movie_name']!=10]


    df['grade10'] = round(df['grade10'],0)
    df[df['grade10']>10] = 10

    for grade in df['grade3'].unique():
        # считаем кол-во 0, которые будет менять
        counts = df[(df['grade10'] != 0) & (df['grade3']==grade)]['grade10'].value_counts(normalize=True)

        # заменяем в зависимости от % других категорий;
        #len(zero_indices)` возвращает количество строк в DataFrame, где `grade10 == 0
        #- `replace=True`: выборка выполняется с заменой, что позволяет выбирать одно и то же значение несколько раз.
        #- `weights=counts`: указывает, что выборка должна быть взвешена
        zero_indices = df[df['grade10'] == 0].index
        replacement_values = counts.sample(len(zero_indices), replace=True, weights=counts, random_state=42).index

        df.loc[zero_indices, 'grade10'] = replacement_values
    return df





def lower_join_tokenizer(st):
    '''
    lower and tokenize input string

    returns: string with tokens separated by space
    '''
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    st=str(st)
    return ' '.join(tokenizer.tokenize(st.lower()))



def lemmatize(st):
    morph_analyzer = pymorphy3.MorphAnalyzer()
    return [morph_analyzer.parse(word)[0].normal_form for word in st.split()]




def full_prep(df, punctuation = False, lemmatized = False, stop_words = False):
    
    print('Удаление особых символов')
    df['content'] = df['content'].progress_apply(lambda st: re.sub(r'\n|\xa0|\nA', ' ', st))
    
    
    df['content'] = df['content'].progress_apply(lambda text: re.sub(r'[^\w\s,.!?]', '', text))
    
    if punctuation:
        print('Удаление знаков препинания')
        df['content'] = df['content'].progress_apply(lambda text: re.sub(r'[^\w\s]', '', text))

    print('Токенизация')
    df['content'] = df['content'].progress_apply(lower_join_tokenizer)

    if  lemmatized:
        print('Лемматизация')
        result = Parallel(n_jobs=8)(delayed(lemmatize)(text) for text in df['content'])
        df['content'] = result

    if stop_words:
      print('Удаление стоп-слов')
      nltk.download('stopwords')
      stop_list = stopwords.words('russian')
      df['content'] = df['content'].progress_apply(lambda x: [item for item in x.split(' ') if item not in stop_list])
    return df

def generate_wordcloud(movie_name, dataframe):
    # Фильтруем рецензии для указанного фильма
    reviews = dataframe[dataframe['movie_name'] == movie_name]['clean_content']
    
    # Объединяем все рецензии в один текст
    text = " ".join(reviews).replace('фильм', '').replace('это', '').replace('который', '').replace('весь', '').replace('всё', '')
    
    # Создаем облако слов
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for '{movie_name}'", fontsize=16)
    plt.show()
