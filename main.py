# unsupervised, find topics or classification labels inherent in dataset

# train on those labels

# inference: new data (from test)

import utils
import clustering

n_keywords = 10

with open('train.txt') as f:
    texts = f.readlines()

total_docs = 10 #len(texts)
texts = texts[:10]
print(total_docs)

freqs = utils.get_dataset_frequencies(texts)
print("here")

summaries = []
for text in texts:
    summaries.append(utils.summarise(text, freqs, total_docs, n_keywords))

for summary in summaries:
    print(summary)

best_params_st1, best_clusters_st1, trials_st1 = clustering.cluster_summaries(summaries)
print(best_params_st1, best_clusters_st1, trials_st1)