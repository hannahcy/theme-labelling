# theme-labelling: Machine Learning Coding Challenge

## How To Run

Use requirements.txt to set up venv or conda, your choice of virtual environment package manager, eg:

    pip install -r requirements.txt

To train a new LDA model:

    python lda.py

This takes about 20 minutes to train on full dataset, but I have provided a trained model etc

To test on an unseen document/s: 

    python lda_inference.py -f (your-test-docs).txt

This assumes the existence of a trained LDA model, a dictionary of vocabulary and a file containing topic names available (all provided in this repo)

Document/s should be in a .txt file, one document per line

    

Alternatively, you can inspect topics.json for the list of topics that were discovered and named when I ran lda.py myself.

## Assumptions and Decisions

Although the '__label__' field in the dataset is known to be unreliable, I took it as an indication of the type of label we might reasonably be looking for from this task (ie general area or topic of the content of each article, as opposed to, for example, length or tone of voice of the article). 

I also used it to estimate a number of labels, and a spread of those labels (while acknowledging they are unreliable, I assumed they might be indicative):

    {
    'Energy&Resources&Utilities': 21786, 
    'Legal&Defence': 6594, 
    'Entertainment': 5220, 
    'Food&Beverage': 17059, 
    'Information,Technology&Telecommunications': 5833, 
    'Pharmaceutical': 9679, 
    'Sport&Gaming': 11199, 
    'Automotive&PetrolServices': 16449, 
    'RealEstate&PropertyServices': 7591, 
    'Agriculture,Forestry&Fishing': 15183, 
    'Education': 13938, 
    'Finance&Insurance&BusinessServices': 26622, 
    'Unions': 1143, 
    'Religion': 2300, 
    'RetailTrade': 8193, 
    'Health&CommunityServices': 20387, 
    'Transport&Storage': 15243, 
    'Hotels&Accommodation': 1143, 
    'Construction&Manufacturing': 7797, 
    'Tourism&Events': 3465, 
    'Media&Publications': 8182, 
    'Cultural&Recreational': 4726, 
    'Environment': 3150
    }

These are somewhat imbalanced, but within an order of magnitude of each other, so I didn't do any preprocessing of the dataset itself (for example, to take a balanced sample from across the topics) -- these labels have also been described as unreliable, so I didn't want to put too much faith in them.

Since I decided to consider this as a topic classification task, I used LDA, which is a well-known approach/solution to this problem. The only problem with LDA is that you need to specify a number of labels/clusters, so I used 23 which is the number of unique labels in the dataset, because this was the easiest evidence I could find to settle on a reasonable number. NUM_TOPICS can be changed and run again if this is inappropriate. If I had time, I would run some more preliminary analysis of the data to refine this number.

I used Bag of Words instead of TF-IDF, because it was easier to implement and my research showed it usually has similar results on this type of task, while BoW often allows more detailed and specific topics. However, with more time I would have experimented with TF-IDF as well and perhaps changed this approach if it had more accurate and specific results.

I chose to train an LDA model using the Gensim library (the other popular library is scikit-learn) because scikit-learn's implementation of LDA is known to be very slow and inefficient. With more time it might be worth trying scikit-learn as well.

I chose to do 10 passes as a balance between accuracy and efficiency. If this was being used for real, I would probably do more passes, or at least experiment with a few different numbers of passes. My research found that 10-50 was a promising range to check, and even just 10 was reasonably time-consuming so I kept it quick for this task!

## Specific Questions from Isentia

### How do you evaluate the accuracy and correctness of your model(s)?

I ran the first 200 documents from test.txt through lda_inference.py and checked all of them (skimmed the article, checked the labels). This was a sanity check, not a proper assessment. For that, we could check Topic Coherence, ie how semantically similar the top keywords within a given topic cluster are. If we're convinced the discovered topics are good ones, we could manually annotate a small set of test articles and test it that way.

### What could you do to improve the accuracy of your model?

We could experiment with more or fewer topics, more passes through the training data, try TF-IDF instead of Bag of Words, or add more (variety of) data. We could try excluding keywords that are under a TF-IDF threshold, rather than relying solely on a predefined list of stopwords. If I knew a bit more about how the labels inside the dataset were generated, and whether we can rely on them to estimate the distribution of topics, we could perhaps balance out the topics in the training set.

### How would you serve your model(s) in production?

I would create a Docker container for this environment and model to live in, then host it on a Cloud service like Azure or AWS -- Cloud services can be better than hosting on a server when the model will only be sparsely used, because you can pay for only what you use. Also depending how this needs to be used in a bigger picture workflow, data could be collected and batch processed offline (for example, if they just want a summary each morning when they start work), or queued to be processed in real time or as soon as possible. In extreme cases, we could deploy it on the client side (edge deployment, for example if we want to tag every news article they navigate to in their browser).

### Imagine that there are future requirements that ask you to add a new class/topic/pattern? What would re-training look like? Are there any concerns? How would you avoid these concerns?

The LDA model can be updated by adding new data, but if we know the number of topics that need to be included, we could retrain the LDA model from scratch and specify this new number of topics. It this involves new data, we would need to add it to the training set and make sure there were a balanced number of examples, ie don't just add one example, or add so many that the dataset is swamped. This is true whether we just update the existing model or retrain it from scratch. The main concern would be that this retraining/updating would cause degradation of performance on the old dataset/domain. The new/updated model could be tested alongside the old model on old data to check for agreement, if we know the previous model to be reliable (this at least ensures backwards compatibility).

# APPENDIX: Instructions from Isentia README:

## Problem Description

As a media monitoring company, we try to make sense of the media that is created every day. One area in particular is written articles, where we want to automatically determine the type of the article. This knowledge helps us to filter articles and present to our customers only what they really care about.

You are given a corpus of text articles, included in this repo. We have split the corpus into test.txt.gzip and train.txt.gzip. This split was made arbitrarily, there is no structural difference between both and you may choose to ignore it if you wish.

Your job is to create a service that classifies the articles into related groups by detecting patterns inside the dataset. When your service is given a new article, your model(s) should return the type(s) that this article belongs to.

The problem is unsupervised. The corpus contains a label field for each article but it is unreliable and you can ignore it.

This could be a text classification problem, or a clustering problem, or a topic detection problem or a combination of these. It is up to you to define. There are no right or wrong answers so long as your approach is properly justified. The important thing is that you demonstrate your ability to translate a vague (but realistic) problem into a machine learning model that gives a reasonable solution, and that you are able to explain the choices that you make in creating a model and solving this task.


## Technical specifications

Your program should be written in Python 3. You can use any other library of your choice that you are familiar with.

You must check in your code in a git repository. You do not have to check in the corpus again, unless you made changes to it.

You can use the full corpus or part of it, if you want to save on computation time. You will not be judged on the accuracy of your final model. What matters is your justification for the approach that you take.

You may use a notebook to explain/solve problems, make graphs or show tables.

## Extra questions

Answers to the following questions should be given in the accompanying readme

- How do you evaluate the accuracy and correctness of your model(s)?
- What could you do to improve the accuracy of your model?
- How would you serve your model(s) in production?
- Imagine that there are future requirements that ask you to add a new class/topic/pattern? What would re-training look like? Are there any concerns? How would you avoid these concerns?


## Evaluation

Your solution will be judged on the following criteria:

- inclusion of a readme with explanation and answers
- completeness of the solution
- clear overall design
- readability of the code
- clear instructions on how to run your solution
- use of git

Bonus

- demo code for serving your model(s).

## Submission

Host your repository online (e.g. on github), make sure that it is public so that we can see it, and tell us the URL.

Make sure your submission includes a readme where you explain your reasoning, the decisions you made, and your answers to the extra questions. Without this documentation we cannot evaluate your code!

