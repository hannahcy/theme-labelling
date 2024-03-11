""" 
Usage: python lda_inference.py -f <name-of-test-file.txt>
This module assumes there is an existing trained LDA model, dictionary of vocabulary and file containing topic names available.
Run lda.py first if these files are not available
"""
import argparse
import json
from operator import itemgetter
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer("english")

import nltk
nltk.download('wordnet')


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


def get_labels(text):
    """ Loads the trained LDA model and runs inference on the new text, then returns the label names and scores for each of the LDA topics over a threshold (0.01) """
    processed_text = preprocess(text)
    lda_model = gensim.models.LdaMulticore.load('lda.model')
    dictionary = gensim.corpora.Dictionary.load('lda-dictionary')
    bow_corpus = dictionary.doc2bow(processed_text)
    doc_lda = lda_model[bow_corpus]
    with open('topics.json', 'r') as f:
        topics = json.load(f)
    labels = []
    for idx, score in doc_lda:
        topic = topics[str(idx)]
        labels.append((topic, score))
    labels = sorted(labels, key=itemgetter(1), reverse=True)
    return labels


def test(test_file):
    """ Reads a test file and, assuming each line is one article to be tested, tests each one """
    with open(test_file) as f:
        test_texts = f.readlines()
    #for text in test_texts[:200]: # uncomment to only test the first 200 documents in test.txt
    for text in test_texts:
        labels = get_labels(text)
        print(text[:100]) # Print first 100 characters to easily search and find the relevant article for spot checking
        print(labels)
        print()


if __name__ == '__main__':
    
        parser = argparse.ArgumentParser()                                               

        parser.add_argument("--file", "-f", type=str, required=True)
        args = parser.parse_args()

        file = args.file

        test(file)

    
    


