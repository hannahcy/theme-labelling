import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import pandas as pd
stemmer = SnowballStemmer("english")

import nltk
nltk.download('wordnet')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

if __name__ == '__main__':
    with open('train.txt') as f:
        texts = f.readlines()

    processed_docs = []

    for doc in texts[:100]:
        processed_docs.append(preprocess(doc))

    dictionary = gensim.corpora.Dictionary(processed_docs)

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    document_num = 0
    bow_doc_x = bow_corpus[document_num]

    #for i in range(len(bow_doc_x)):
    #    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
    #                                                    dictionary[bow_doc_x[i][0]], 
    #                                                    bow_doc_x[i][1]))
        
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics = 10, 
                                    id2word = dictionary,                                    
                                    passes = 10,
                                    workers = 2)

    topics = {}
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        print(type(topic))

