# **Sentiment Classification For Social Media (Twitter Data)**

For this project, we will be using the "SemEval 2017 task 4" corpus provided on the module website, available through the following [link](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs918/semeval-tweets.tar.bz2). We will focus particularly on Subtask A, i.e. classifying the overall sentiment of a tweet as positive, negative or neutral.
For quick acces, start by running the provided notebook for analysing the sentiment analysis of Twitter data using Naïve Bayes, SVM and LSTM classifiers: 

- ```sentiment_analysis_twitter.ipynb``` 

The below are the steps we follow to achieve our goal.
## **Data Cleaning:**
For each dataset including training, validation and test data, the input is read by splitting each line by ‘\t’. tweets, tweetids, and sentiments, are captured in different dictionaries. Each dictionary has the dataset location as the key and values to be an array of tweets, tweetids, and sentiments respectively.  
Preprocessing:
While each input is being read, the tweets are preprocessed by going through the following steps
1.	***Regex***
    * URL removal: Different kinds of url are available. For example,
‘https://www.abc.com’, ‘www.abc.com’.
These are carefully removed as it does not hold any useful information for our classification.

    * HTML entities removal: Entities like &lg; &amp; are commonly used to denote emojis in tweet data. These are removed using the following regex,
re.sub(r"&[a-zA-Z0-9]+;", "",processed)

    * @user mentions: User mentions are common in tweet data and they usually hold the userid of the person mentioned. These are also removed.
re.sub(r"@\w+","",processed)

    * Alphanumeric removal: Alpha-numeric values are removed except for the space, as usually these are mostly user ids
 re.sub(r"[^A-Za-z0-9 ]","",processed)

    * White space removal: tweets often contain an uneven number of spaces due to human errors, so these are removed too. 
re.sub(r"\s+"," ",processed) 

    * Numbers removal: Numbers are also removed as they do not convey any emotion. 
re.sub(r"\b[0-9]+\b","",processed)

2.	***Stop word Removal***

    Stop words are commonly used in the language and they often do not hold any significant meaning. These are conjunctions, prepositions like “the”, “an”, “of”, “on”, “this”, “that”, etc. These are removed with the help of remove_stopwords() as they do not contribute to the classification and they skew the data. Removing them helps in dimensionality reduction.

3.	***POS Tagging***
    Pos tagging is to tag each word to its parts of speech. It is used to analyse the sentiment polarity of the words in the sentence/text. Each word in the tweet is assigned a pos tag using nltk. pos_tags() to help map it to get more context and meaning of the text.

4.	***Lemmatizing***

    Lemmatization is performed in lemmatize(), to break a word down to its root meaning. For example, “running” and “ran” will have the root word “run”. This is applied to reduce the number of words that contains the same root word.

## **Feature Extraction:**

For ML models, we need to convert the words into vectorized values. The following three approaches are used for feature extraction.

1.	***Bag of Words:*** It represents each tweet as a vector of word frequencies. But it does not capture the context and meaning of words

2.	***Term Frequency-Inverse Document Frequency:*** It represents each tweet as a weighted word frequency vector. These weights reflect the significance of words in the tweet and corpus of documents. 

3.	***Word embeddings:*** Word embeddings are used vectors to represent words in higher dimensions to capture the semantics which is not captured in the bag of words. This is the state-of-the-art approach for capturing word similarity. Glove embedding is used in LSTM classifier, which is based on the idea that the meaning of a word can be examined by the co-occurrence of words in a corpus. It works by measuring the difference between the dot product of word embeddings and its co-occurrence probabilities.

## **Model Selection:**

Below are the classifiers considered and implemented for the classification

1.	***SVM:*** Text classification involves large feature spaces and SVM is effective in handling high-dimensional feature spaces. Our data is a bit imbalanced for neutral tweet data and SVM is known to handle it well. SVM is used as one of the three classifiers with hyperparameter C set to 0.1 and 1 for bag of word vectorizing and tf-idf vectorizing respectively. This is determined using GridSearch hyperparameter optimization.

2.	***Naïve-Bayes:*** One other traditional text classification algorithm is MultiNomial Naïve Bayes. It models the probabilistic distribution of word frequencies of a class. Multinomial Naïve Bayes is used with hyperparameter alpha set to 0.1 and 1 for bag of word vectorizing and tf-idf vectorizing respectively.

3.	***LSTM:*** A neural network architecture is created which involves the following steps.
    * Creating a vocabulary to get all the unique words in the tweet data and creating a word index with words as keys and unique integers as values for reference.
    * Downloading and reading glove data and creating a dictionary with words as keys and vectors as values.
    * Creating an embedding matrix by mapping every word in the word_index dictionary to the corresponding array in the glove vector.
    * Preparing training and test datasets using word_index and loading it in Data Loader for feeding this data to our neural network. For unseen words, we are setting a default index/key in word_index <unk> and using its value. Index 0 is used as a placeholder and no values are assigned. Here, the maximum length for a document is set to 30 and if the word length of the tweet is lesser than this length then it is padded by 0. An additional step of replacing all the word_index values over 5000 to 0 is done for all the words as this is required for the next step. Y labels are mapped to 0 for positive, 1 for negative, and 2 for neutral as this is needed for the next step.
    * A LSTM architecture is defined with four layers, an embedding layer of dimension 5000*100, a lstm layer of size 100*256 and a linear layer of size 64*1 and a dropout layer to avoid overfitting of data by selecting a random set of values and setting it to be 0. Since the linear layer itself produces a continuous output, a softmax activation is used to produce class probabilities in the prediction step.
    * Hyperparameters are tested and a training loop is created where the model is continuously trained, the loss is calculated, and backward propagation is performed to determine updated weights and it is optimized. The gradients are reset after every iteration to avoid corrupted weights.
    * Model is then run with test data to obtain predictions. In the last block of code, the model is instantiated with hyperparameters and fed with a dataset and trained. The predictions are then converted back to get the original labels and evaluated.

## **Hyper Parameter Optimization:**

The parameters for all of the classifiers are found using GridSearch method over the values which are as follows.

'tfidf__max_df': [0.5, 0.75, 1.0],
'tfidf__max_df': [0.5, 0.75, 1.0],
'tfidf__ngram_range': [(1, 1), (1, 2)],
'nb__alpha': [0.001,0.0001,0.1, 1.0, 10.0],
'clf__C': [0.1, 1, 10, 100, 1000], 

This piece of code is commented out as this will need additional run time.
Each of the classifiers is trained against a different feature extraction method and a set of hyperparameters. 

## **Regularization:**

1.	Neural network layer dropout is done by setting half of the values to zero
2.	Batch size is set to 64 which is lesser to make it generalize
3.	Loss function is changed after a few epochs to make the task difficult for the model to classify and hence it learns better.
4.	Learning rate is also reduced by a factor every few epochs
These steps will help generalize the data classification and prevent overfitting.

## **Performance:**

* Out of these Linear SVM with tf-idf vectorizer gives ~0.6 as macro averaged F1 value, which is the highest amongst all.
* Like SVM with the Bag of words, Naive Bayes with the Bag of words also gives around ~0.5 as the F1 value. There is only a slight difference in the difference between the different vectorizers for both SVM and   Bayes.
* LSTM gives good performance with word embeddings too. All of these classifiers perform almost the same around ~0.5 or 0.6 but the misclassified instances in the confusion matrix say that there is room for efficiency. LSTM could be run for more epochs to converge to a better state with high-powered computers. 

## **Results:**

### **Training naive-bayes**
semeval-tweets\twitter-dev-data.txt (bow-naive-bayes): 0.595 

**Confusion Matrix** \
&ensp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; positive  negative  neutral \
positive&nbsp; &nbsp;    0.609&nbsp;&nbsp; &nbsp;     0.080&nbsp;&nbsp; &nbsp;     0.311     
negative&nbsp;    0.058&nbsp;&nbsp; &nbsp;     0.606&nbsp;&nbsp; &nbsp;     0.335     
neutral&nbsp; &nbsp;     0.227&nbsp;&nbsp; &nbsp;     0.141&nbsp;&nbsp; &nbsp;     0.632     

### **Training naive-bayes**
semeval-tweets\twitter-dev-data.txt (tf-idf-naive-bayes): 0.542 

**Confusion Matrix** \
&ensp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; positive  negative  neutral \
positive&nbsp; &nbsp;    0.610&nbsp;&nbsp; &nbsp;     0.081 &nbsp;&nbsp; &nbsp;    0.309     
negative&nbsp;    0.063&nbsp;&nbsp; &nbsp;     0.656&nbsp;&nbsp; &nbsp;     0.281     
neutral&nbsp; &nbsp;     0.249&nbsp;&nbsp; &nbsp;     0.162&nbsp;&nbsp; &nbsp;     0.589     

### **Classifier SVM**
#### **Training svm**
semeval-tweets\twitter-dev-data.txt (bow-svm): 0.596 \
**Confusion Matrix**  
&ensp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; positive  negative  neutral \
positive&nbsp; &nbsp;    0.677&nbsp;&nbsp; &nbsp;     0.054&nbsp;&nbsp; &nbsp;     0.269     
negative&nbsp;    0.073&nbsp;&nbsp; &nbsp;     0.629&nbsp;&nbsp; &nbsp;     0.297     
neutral &nbsp; &nbsp;    0.228&nbsp;&nbsp; &nbsp;     0.153&nbsp;&nbsp; &nbsp;     0.619     

#### **Training svm**
semeval-tweets\twitter-dev-data.txt (tf-idf-svm): 0.609 

**Confusion Matrix**
&ensp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; positive  negative  neutral \
positive&nbsp;&nbsp;  0.663&nbsp;&nbsp; &nbsp;        0.051 &nbsp; &nbsp;     0.287     \
negative&nbsp;  0.051&nbsp;&nbsp; &nbsp;       0.629 &nbsp; &nbsp;     0.321      \
neutral &nbsp;&nbsp;  0.239 &nbsp;&nbsp; &nbsp;      0.144  &nbsp; &nbsp;    0.617      

### **Training LSTM**
100%|██████████| 1410/1410 [05:40<00:00,  4.14it/s] \
Epoch  0  - Train Loss : 1.007 \
Valid Loss : 0.939 \
Valid Acc  : 0.519 \
100%|██████████| 1410/1410 [05:34<00:00,  4.22it/s] \
Epoch  1  - Train Loss : 0.844 \
Valid Loss : 0.801 \
Valid Acc  : 0.627 \
100%|██████████| 1410/1410 [05:37<00:00,  4.18it/s] \
Epoch  2  - Train Loss : 0.782 \
Valid Loss : 0.770 \
Valid Acc  : 0.653 \
100%|██████████| 1410/1410 [06:06<00:00,  3.85it/s] \
Epoch  3  - Train Loss : 0.744 \
Valid Loss : 0.760 \
Valid Acc  : 0.659 \
100%|██████████| 1410/1410 [05:38<00:00,  4.17it/s] \
Epoch  4  - Train Loss : 0.701 \
Valid Loss : 0.767 \
Valid Acc  : 0.659 \
100%|██████████| 1410/1410 [05:55<00:00,  3.97it/s] \
Epoch  5  - Train Loss : 0.650 \
Valid Loss : 0.800 \
Valid Acc  : 0.641 \
100%|██████████| 1410/1410 [07:18<00:00,  3.22it/s] \
Epoch  6  - Train Loss : 0.590 \
Valid Loss : 0.842 \
Valid Acc  : 0.647 \
100%|██████████| 1410/1410 [05:51<00:00,  4.01it/s] \
Epoch  7  - Train Loss : 0.525 \
Valid Loss : 0.915 \
Valid Acc  : 0.627 \
100%|██████████| 1410/1410 [05:32<00:00,  4.24it/s] \
Epoch  8  - Train Loss : 0.464 \
Valid Loss : 0.982 \
Valid Acc  : 0.620 \
100%|██████████| 1410/1410 [05:32<00:00,  4.24it/s] \
Epoch  9  - Train Loss : 0.415 \
Valid Loss : 1.051 \
Valid Acc  : 0.618 


## Discussion on **LSTM :**
To understand the learning progress of the model, for 10 epochs, the loss and the accuracy of the validation set are observed. For the initial few epochs, the accuracy kept growing but after epoch 3 the model started overfitting the data and hence the accuracy decreased moving down, this can be seen from the image below. To let the model learn more about the data it was allowed to overfit first. It was then run with lower batch sizes and changed loss functions to make it harder for the model to learn, to generalize.
 
   
