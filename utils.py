""" Utility functions for Topic/Theme Analysis challenge """

import spacy
import json
import os.path
import nltk
from nltk.corpus import stopwords
import numpy as np
import heapq

nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
#print(STOPWORDS)

def clean_text(raw_doc):
    cleaned_doc = raw_doc[:'__']


def tokenize(raw_doc):
    """ Use spacy to tokenize and process a given text """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_doc)
    return doc


def ignore_token(token):
    return token.is_punct or token.lemma_ in STOPWORDS or token.pos_ == 'PROPN' or token.lemma_.startswith('label')


def get_dataset_frequencies(docs):
    """ Returns a dictionary of every (lemmatised) word in the dataset, either by reading an existing file or by counting """

    freqs_file = 'freqs.json'

    if os.path.isfile(freqs_file):
        print("File found:", freqs_file)
        with open(freqs_file, 'r') as f:
            return json.load(f)
    
    freqs = {}
    
    for raw_doc in docs:
        tokens = tokenize(raw_doc.lower())
        lemmas_counted = []
        for token in tokens:
            lemma = token.lemma_
            if ignore_token(token) or lemma in lemmas_counted:
                continue
            if lemma in freqs:
                freqs[lemma] += 1
            else:
                freqs[lemma] = 1
            lemmas_counted.append(lemma)

    freqs = {lemma: count for lemma, count in sorted(freqs.items(), key=lambda item: item[1])}

    with open(freqs_file, 'w') as f:
        json.dump(freqs, f)

    return freqs


def get_document_frequencies(doc):
    """ Returns a dictionary of every (lemmatised) word in a given document """
    freqs = {}
    for token in doc:
        if ignore_token(token):
            continue
        lemma = token.lemma_
        if lemma in freqs:
            freqs[lemma] += 1
        else:
            freqs[lemma] = 1
    return freqs


def term_frequency(doc, freqs, lemma):
    """ """
    return freqs[lemma] / len(doc)


def inverse_document_frequency(lemma, freqs, total_docs):
    """ """
    try:
        word_occurrence = freqs[lemma] + 1
    except:
        word_occurrence = 1
    return np.log(total_docs / word_occurrence)


def tf_idf(document, freqs, total_docs):
    """ """
    doc = tokenize(document.lower())
    doc_freqs = get_document_frequencies(doc)
    tf_idfs = {}
    for token in doc:
        if ignore_token(token):
            continue
        lemma = token.lemma_
        tf = term_frequency(doc, doc_freqs, lemma)
        idf = inverse_document_frequency(lemma, freqs, total_docs)
        tf_idfs[lemma] = tf * idf
    tf_idfs = {lemma: score for lemma, score in sorted(tf_idfs.items(), key=lambda item: item[1])}
    return tf_idfs


def summarise(text, freqs, total_docs, n_keywords):
    tf_idfs = tf_idf(text, freqs, total_docs)

    keywords = heapq.nlargest(n_keywords, tf_idfs.items(), key=lambda i: i[1])
    return [keyword[0] for keyword in keywords]