# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:46:27 2020

@author: zyyzy
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import re
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")

#import data
def read_data(x, y):
    files = os.listdir("{}/{}".format(x, y))
    content = []
    for filename in files:
        with open("{}/{}/{}".format(x, y, filename), 'r', encoding = 'UTF-8') as f:
            content.append(f.read())
    content = pd.DataFrame(content, columns =['review'])
    if y == 'pos':
        content['label'] = 1
    else:
        content['label'] = 0
    return content
trainPos = read_data("train", "pos")
trainNeg = read_data("train", "neg")
testPos = read_data("test", "pos")
testNeg = read_data("test", "neg")

#Since too many data make the tensorflow model very slow, I only choose part of data to train.
train = pd.concat([trainPos.loc[0:199], trainNeg.loc[0:199]], ignore_index= True)

#Clean the comments
def review_to_words(raw_review):
        #Delete the punctuation marks and other unexpected marks.
        letters_only= re.sub("[^a-zA-Z]"," ",raw_review)
        #Turn the capital letters into lower case.
        words= letters_only.lower().split()
        #Delete stopwords.
        stops= set(stopwords.words("english")) 
        meaningful_words= [w for w in words if not w in stops]

        return(" ".join(meaningful_words))

#Use the function clean the train data.
num_reviews= train.shape[0]
cleanReviews= []
for i in range(0, num_reviews):
        cleanReviews.append(review_to_words(train["review"][i]))

#Use set to avoid the repetitive words.
s = set()
comments = cleanReviews
labels = train['label']
#Split the sentence into seperated words.
for i in tqdm(range(len(comments))):
    comment = comments[i]
    text = comment.split(' ')
    s.update(text)

word_numpy = np.array(list(s))

#Create the dictionary including all words in the train comments.
dataDict = {}
dataDict["start"] = 1
dataDict["new_word"] = 2
count = 3

#Assign values to the words in the dictionary.
for w in word_numpy:
    dataDict[w] = count
    count += 1

#This function aims to turn one comment into a vector form by looking up the dictionary.
def word2vector(comment):
    global dataDict
    #Split the comment.
    texts = comment.split(' ')
    #All sentences begin with the <START> in the dictionary
    temp = np.array([1])
    
    for text in texts:
        if(dataDict.get(text) != None):
            temp = np.append(temp, dataDict[text])
        else:
            #If a word can't be found in the dictionary, it will be marked as <UNK> by number 2.
            temp = np.append(temp, 2)
    return temp

#Process all train data.
trainReview = []
for i in tqdm(range(len(comments))):
    comment = comments[i]
    comment2 = word2vector(comment)
    trainReview.append(comment2)
#trainReview = np.array(trainReview)

#Check the max length of the comments.
temp = []
for i in tqdm(range(len(trainReview))):
    a = len(trainReview[i])
    temp.append(a)
max(temp)

#By adding number 0 or truncating the excessive words, ensure all the comment vectors have same the length.
trainReview = keras.preprocessing.sequence.pad_sequences(trainReview,
                                                           value = 0,
                                                           padding = 'post',
                                                           truncating = 'post',
                                 #Actually the max length is more than 2000, but I truncate many vectors to 100 for the fitting speed.
                                                           maxlen = 100)

#Disorder the data.
seed = np.random.permutation(labels.shape[0])
trainReview = trainReview[seed, :]
labels = labels[seed]


#Fit the sequential model.
trainReview = K.cast_to_floatx(trainReview)
labels = K.cast_to_floatx(labels)

model = keras.Sequential()
model.add(keras.layers.Embedding(20000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(512, activation = tf.nn.relu))
#The sigmoid function ensure the outcome is 0 or 1.
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))

#Define a optimizer.
adam = keras.optimizers.Adam(0.0001)

#Compile the model.
model.compile(optimizer=adam,
              metrics=['acc'],
              loss='binary_crossentropy')

#model.summary()

#Define a function to monitor the process while fitting.
tensorBoard = keras.callbacks.TensorBoard(log_dir = 'keraslog')

#Fit the model
model.fit(trainReview, labels, batch_size=128, validation_split=0.2, epochs=50, callbacks=[tensorBoard], verbose=2)



test = testPos.loc[183:284]
num_reviews= test["review"].size
cleanReviews= []
for i in range(0, num_reviews):
        cleanReviews.append(review_to_words(trainPos["review"][i]))

comments = cleanReviews

testReview = []
for i in tqdm(range(len(comments))):
    comment = comments[i]
    comment2 = word2vector(comment)
    testReview.append(comment2)

testReview = keras.preprocessing.sequence.pad_sequences(testReview,
                                                           value = 0,
                                                           padding = 'post',
                                                           truncating = 'post',
                                 #Actually the max length is more than 2000, but I truncate many vectors to 100 for the fitting speed.
                                                           maxlen = 100)
consequence = []
for i in tqdm(range(len(testReview))):
    temp = np.expand_dims(testReview[i], axis = 0)
    a = model.predict(temp)
    if(a[0][0] >=0.5):
        consequence.append(1)
    else:
        consequence.append(0)


