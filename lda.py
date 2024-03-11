""" 
Usage: python lda.py
This module assumes the existence of a 'train.txt' file containing all the training data.
You need to add an OpenAI api_key, which I can provide in an email if you don't have one (if you run this code, it will only make 23 short calls to GPT-4 so not expensive)
"""
import json
import openai
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer("english")

import nltk
nltk.download('wordnet')

OPEN_AI_KEY = "<YOUR_API_KEY>"
NUM_TOPICS = 23 # This was obtained from inspection of the train.txt dataset


def lemmatize_stemming(text):
    """ Returns the text (a token) after lemmatizing and stemming """
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    """ Tokenises the text, filters out stopwords and words 3 characters or shorter, lemmatises and stems each remaining token """
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


def get_label_name(list_of_keywords):
    """ From the list of keywords that LDA has specified for a given topic, ask GPT-4 to give that topic a meaningful name """
    keywords_as_string = ''.join(list_of_keywords)
    openai.api_key = OPEN_AI_KEY

    prompt="Define a generic label for a topic or theme that could occur in a news media article that contains the following keywords "+keywords_as_string+". The name for this topic should be 1-3 words, separated by an underscore. The topic name should be generic enough that it could be applied to many articles, so generalise away from any specifics"
    response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

    label_name = response.choices[0].message.content
    return label_name


if __name__ == '__main__':
    with open('train.txt') as f:
        texts = f.readlines()

    processed_docs = []
    for doc in texts:
        processed_docs.append(preprocess(doc))

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.save('lda-dictionary')

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics = NUM_TOPICS, 
                                    id2word = dictionary,                                    
                                    passes = 10,
                                    workers = 2)
    
    lda_model.save('lda.model')

    topics = {}
    for idx, topic in lda_model.print_topics(-1):
        l = topic.split('"')[1::2]
        label_name = get_label_name(l)
        print(l) # I have left these in here so you can see the top keywords and the label that GPT-4 chose for those keywords
        print(label_name)
        topics[idx] = label_name
    
    with open('topics.json', 'w') as f:
        json.dump(topics, f)

