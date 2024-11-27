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

tqdm.pandas()    





def lower_join_tokenizer(st):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    st=str(st)
    return ' '.join(tokenizer.tokenize(st.lower()))

def lemmatize(st):
    morph_analyzer = pymorphy3.MorphAnalyzer()
    return [morph_analyzer.parse(word)[0].normal_form for word in st.split()]


def preprocessing(df, lemmatized = False, stop_words = False):
    print('Удаление особых символов')
    df['content'] = df['content'].progress_apply(lambda st: re.sub(r'\n|\xa0|\nA', ' ', st))
    df['content'] = df['content'].progress_apply(lambda text: re.sub(r'[^\w\s,.!?]', '', text))
    
    print('Токенизация')
    df['content'] = df['content'].progress_apply(lower_join_tokenizer)

    if  lemmatized:
        print('Лемматизация')
        result = Parallel(n_jobs=8, verbose=10)(delayed(lemmatize)(text) for text in df['content'])
        df['content'] = result

    if stop_words:
      print('Удаление стоп-слов')
      nltk.download('stopwords')
      stop_list = stopwords.words('russian')
      df['content'] = df['content'].apply(lambda x: [item for item in x if item not in stop_list])
    return df