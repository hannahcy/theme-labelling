# Copied and modified from  https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import pandas as pd
import tqdm as t
import random
import umap
import hdbscan
import numpy as np
from hyperopt import hp, tpe, Trials, partial, space_eval, fmin, STATUS_OK
import hyperopt
import spacy
import collections
from datetime import datetime
import chatintents
from chatintents import ChatIntents
import re
import unicodedata
import emoji



def embed(model, model_type, sentences):
  """
  wrapper function for generating message embeddings
  """
  
  if model_type == 'use':
    embeddings = model(sentences)
  elif model_type == 'sentence transformer':
    embeddings = model.encode(sentences)
        
  return embeddings

def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state)
                            .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters

def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost

def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times 
    and return a summary of the results
    """
    
    results = []
    
    for i in t.trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size, 
                                     random_state = 42)
    
        label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
                
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, cost])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'min_cluster_size', 'label_count', 'cost'])
    
    return result_df.sort_values(by='cost')


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize, which incorporates constraints
    on the number of clusters we want to identify
    """
    
    clusters = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 random_state = params['random_state'])
    
    label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
    
    #15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15 
    else:
        penalty = 0
    
    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayseian search on hyperopt hyperparameter space to minimize objective function
    """
    
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=max_evals, 
                trials=trials)

    best_params = space_eval(space, best)
    print ('best:')
    print (best_params)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters = generate_clusters(embeddings, 
                                      n_neighbors = best_params['n_neighbors'], 
                                      n_components = best_params['n_components'], 
                                      min_cluster_size = best_params['min_cluster_size'],
                                      random_state = best_params['random_state'])
    
    return best_params, best_clusters, trials


def most_common(lst, n_words):
        """
        Return most common n words in list of words
        Arguments:
            lst: list of words
            n_words: int, number of top words by frequency to return
        Returns:
            counter.most_common(n_words): a list of the n most common elements
                                          and their counts from the most
                                          common to the least
        """

        counter = collections.Counter(lst)

        return counter.most_common(n_words)


def extract_labels(category_docs):
    """
    Extract labels from documents in the same cluster by concatenating
    most common verbs, ojects, and nouns
    """

    verbs = []
    dobjs = []
    nouns = []
    adjs = []
    
    verb1 = ''
    verb2 = ''
    verb3 = ''
    adj1 = ''
    adj2 = ''
    dobj1 = ''
    dobj2 = ''
    noun1 = ''
    noun2 = ''
    noun3 = ''
    noun4 = ''

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading language model for the spaCy dependency parser\n"
              "(only required the first time this is run)\n")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # for each document, append verbs, dobs, nouns, and adjectives to 
    # running lists for whole cluster
    for i in range(1, len(category_docs)):
        #print(i, category_docs[i])
        doc = nlp(str(category_docs[i]))
        for token in doc:
            if token.is_stop==False:
                if token.dep_ == 'ROOT':
                    verbs.append(token.lemma_.lower())

                elif token.dep_=='dobj':
                    dobjs.append(token.lemma_.lower())

                elif token.pos_=='NOUN':
                    nouns.append(token.lemma_.lower())
                    
                elif token.pos_=='ADJ':
                    adjs.append(token.lemma_.lower())
    
    # take most common words of each form
    if len(verbs) > 0:
        verb1 = most_common(verbs, 1)[0][0]

    if len(set(verbs)) > 1:
        verb2 = most_common(verbs, 2)[1][0]

    if len(set(verbs)) > 2:
        verb3 = most_common(verbs, 3)[2][0]

    if len(dobjs) > 0:
        dobj1 = most_common(dobjs, 1)[0][0]

    if len(set(dobjs)) > 1:
        dobj2 = most_common(dobjs, 2)[1][0]
    
    if len(nouns) > 0:
        noun1 = most_common(nouns, 1)[0][0]
    
    if len(set(nouns)) > 1:
        noun2 = most_common(nouns, 2)[1][0]

    if len(set(nouns)) > 2:
        noun3 = most_common(nouns, 3)[2][0]
    
    if len(set(nouns)) > 3:
        noun4 = most_common(nouns, 4)[3][0]
    
    if len(adjs) > 0:
        adj1 = most_common(adjs, 1)[0][0]
    
    if len(set(adjs)) > 1:
        adj2 = most_common(adjs, 2)[1][0]
    
    # concatenate the most common verb-dobj-noun-noun (if they exist)
    label_1_words = [verb1, adj1, dobj1]
    
    for word in [noun1, noun2]:
        if word not in label_1_words:
            label_1_words.append(word)
    
    if '' in label_1_words:
        label_1_words.remove('')
    
    label_1 = '_'.join(label_1_words)

    # concatenate a second combination of verb-dobj-noun-noun (if they exist)
    label_2_words = [verb1, adj2, dobj2]
    
    for word in [noun1, noun3]:
        if word not in label_2_words:
            label_2_words.append(word)
    
    if '' in label_2_words:
        label_2_words.remove('')
    
    label_2 = '_'.join(label_2_words)

    # concatenate a third combination of verb-dobj-noun-noun (if they exist)
    label_3_words = [verb2, adj1, dobj1]
    
    for word in [noun2, noun4]:
        if word not in label_3_words:
            label_3_words.append(word)
    
    if '' in label_3_words:
        label_3_words.remove('')
    
    label_3 = '_'.join(label_3_words)

    # concatenate a fourth combination of verb-dobj-noun-noun (if they exist)
    label_4_words = [verb2, adj2, dobj2]
    
    for word in [noun3, noun4]:
        if word not in label_4_words:
            label_4_words.append(word)
    
    if '' in label_4_words:
        label_4_words.remove('')
    
    label_4 = '_'.join(label_4_words)

    # concatenate a fifth combination of verb-dobj-noun-noun (if they exist)
    label_5_words = [verb1, adj2, dobj1]
    
    for word in [noun3, noun4]:
        if word not in label_5_words:
            label_5_words.append(word)
    
    if '' in label_5_words:
        label_5_words.remove('')
    
    label_5 = '_'.join(label_5_words)

    # concatenate a sixth combination of verb-dobj-noun-noun (if they exist)
    label_6_words = [verb2, adj1, dobj2]
    
    for word in [noun1, noun2]:
        if word not in label_6_words:
            label_6_words.append(word)
    
    if '' in label_6_words:
        label_6_words.remove('')
    
    label_6 = '_'.join(label_6_words)

    # concatenate a seventh combination of verb-dobj-noun-noun (if they exist)
    label_7_words = [verb3, adj1, dobj1]
    
    for word in [noun1, noun3]:
        if word not in label_7_words:
            label_7_words.append(word)
    
    if '' in label_7_words:
        label_7_words.remove('')
    
    label_7 = '_'.join(label_7_words)

    # concatenate an eighth combination of verb-dobj-noun-noun (if they exist)
    label_8_words = [verb3, adj2, dobj1]
    
    for word in [noun2, noun4]:
        if word not in label_8_words:
            label_8_words.append(word)
    
    if '' in label_8_words:
        label_8_words.remove('')
    
    label_8 = '_'.join(label_8_words)

    # concatenate an ninth combination of verb-dobj-noun-noun (if they exist)
    label_9_words = [verb3, adj1, dobj2]
    
    for word in [noun1, noun4]:
        if word not in label_9_words:
            label_9_words.append(word)
    
    if '' in label_9_words:
        label_9_words.remove('')
    
    label_9 = '_'.join(label_9_words)

    labels = label_1 + '\n' + label_2 + '\n' + label_3 + '\n' + label_4 + '\n' + label_5 + '\n' + label_6 + '\n' + label_7 + '\n' + label_8 + '\n' + label_9
    
    return labels

def get_group(df, category_col, category):
        """
        Return single category of documents with known labels
        Arguments:
            df: pandas dataframe of documents and associated ground truth
                labels
            category_col: str, name of column with document labels
            category: str, single document label of interest
        Returns:
            single_category: pandas dataframe with only documents from a
                             single category of interest
        """

        single_category = (df[df[category_col] == category]
                           .reset_index(drop=True)
                           )

        return single_category

def apply_and_summarize_labels(df_data, best_clusters, name):
        """
        Assign groups to original documents and provide group counts
        Arguments:
            df_data: pandas dataframe of original documents of interest to
                     cluster
        Returns:
            df_summary: pandas dataframe with model cluster assignment, number
                        of documents in each cluster and derived labels
            labeled_docs: pandas dataframe with model cluster assignment and
                          associated dervied label applied to each document in
                          original corpus
        """

        # create a dataframe with cluster numbers applied to each doc
        category_col = 'label_' + name
        df_clustered = df_data.copy()
        df_clustered[category_col] = best_clusters.labels_

        numerical_labels = np.unique(df_clustered[category_col])
        #print(numerical_labels)
        #print("df_data shape: ", df_data.shape)
        #print("df_clustered shape: ", df_clustered.shape)
        #print(df_clustered.head(500))

        # create dictionary mapping the numerical category to the generated
        # label
        label_dict = {}
        for label in numerical_labels:
            current_category = list(get_group(df_clustered, category_col,
                                                    label)[0])
            #print(current_category)
            label_dict[label] = extract_labels(current_category)

        # create summary dataframe of numerical labels and counts
        df_summary = (df_clustered.groupby(category_col).count()
                      .reset_index()
                      .rename(columns={0: 'count'})
                      .sort_values('count', ascending=False))

        # apply generated labels
        df_summary['label'] = df_summary.apply(lambda x:
                                               label_dict[x[category_col]],
                                               axis=1)

        labeled_docs = pd.merge(df_clustered,
                                df_summary[[category_col, 'label']],
                                on=category_col,
                                how='left')

        return df_summary, labeled_docs


def merge_to_dataframes(labeled_docs, gists, goals, programs, weeks, sessions):
    new_column_gists = pd.DataFrame({'gist': gists})
    new_column_goals = pd.DataFrame({'goal': goals})
    new_column_programs = pd.DataFrame({'program': programs})
    new_column_weeks = pd.DataFrame({'week': weeks})
    new_column_sessions = pd.DataFrame({'session_in_week': sessions})
    labeled_docs = labeled_docs.merge(new_column_gists, left_index=True, right_index=True)
    labeled_docs = labeled_docs.merge(new_column_goals, left_index=True, right_index=True)
    labeled_docs = labeled_docs.merge(new_column_programs, left_index=True, right_index=True)
    labeled_docs = labeled_docs.merge(new_column_weeks, left_index=True, right_index=True)
    labeled_docs = labeled_docs.merge(new_column_sessions, left_index=True, right_index=True)
    return labeled_docs



def cluster_summaries(summaries):

    # define the document embedding models to use for comparison
    #module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    #model_use = hub.load(module_url)
    model_st1 = SentenceTransformer('all-mpnet-base-v2')
    #model_st2 = SentenceTransformer('all-MiniLM-L6-v2')
    #model_st3 = SentenceTransformer('paraphrase-mpnet-base-v2')

    # generate embeddings for each model
    #embeddings_use = embed(model_use, 'use', all_intents)
    embeddings_st1 = embed(model_st1, 'sentence transformer', summaries)
    #embeddings_st2 = embed(model_st2, 'sentence transformer', all_intents)
    #embeddings_st3 = embed(model_st3, 'sentence transformer', all_intents)

    #embeddings = [embeddings_use, embeddings_st1, embeddings_st2, embeddings_st3]

    #for embedding in embeddings:
    #    print(embedding.shape)
    #print(len(embeddings_use))

    #random_use = random_search(embeddings_use, space, 100)
    #print(random_use.head(25))

    hspace = {
        "n_neighbors": hp.choice('n_neighbours', range(3,16)),
        "n_components": hp.choice('n_components', range(3,16)),
        "min_cluster_size": hp.choice('min_cluster_size', range(2,16)),
        "random_state": 42
    }

    label_lower = 80
    label_upper = 150
    max_evals = 1000

    #best_params_use, best_clusters_use, trials_use = bayesian_search(embeddings_use, 
    #      space=hspace,label_lower=label_lower, label_upper=label_upper, max_evals=max_evals)

    best_params_st1, best_clusters_st1, trials_st1 = bayesian_search(embeddings_st1,
        space=hspace,label_lower=label_lower, label_upper=label_upper, max_evals=max_evals)

    #best_params_st2, best_clusters_st2, trials_st2 = bayesian_search(embeddings_st2, 
        #space=hspace,label_lower=label_lower, label_upper=label_upper, max_evals=max_evals)

    #best_params_st3, best_clusters_st3, trials_st3 = bayesian_search(embeddings_st3, 
        #space=hspace,label_lower=label_lower, label_upper=label_upper, max_evals=max_evals)

    #df_summary, labeled_docs = apply_and_summarize_labels(all_intents_df, best_clusters_st1, 'st1')

    return best_params_st1, best_clusters_st1, trials_st1

