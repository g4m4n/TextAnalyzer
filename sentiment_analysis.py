from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from  keras.datasets import imdb
from sklearn.preprocessing import MultiLabelBinarizer

import keras.utils
import numpy as np
import pandas as pd
import csv
import nltk
import itertools
from bs4 import BeautifulSoup
import re


# Embedding
max_features = 20000
maxlen = 140
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

'''
Primero leemos el csv
dataset=np.loadtxt("betsentiment-ES-tweets-sentiment-teams.csv",delimiter=",")
Primer intento con numpy fracaso

#train_set = pd.read_csv("betsentiment-ES-tweets-sentiment-teams.csv")

with open('betsentiment-ES-tweets-sentiment-teams.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        print(row)
'''

def transform_instance(row):
    cur_row = []
    #Prefix the index-ed label with __label__
    ##label = "__label__" + row[0]
    #cur_row.append(row)
    #Clean tweet and tokenize it
    #cur_row.extend( nltk.word_tokenize(row[2].lower()))
    tokenizer = Tokenizer(num_words=140)
    sequences = tokenizer.texts_to_sequences(row)
    row = sequences

def clean_row(tweet):
    tweet = tweet.replace("’", "'")
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
    tweet = tweet.lower()
    return tweet

def preprocess(input_file, output_file, keep=1,filas=0):
    with open(output_file, 'w') as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=',', lineterminator='\n')
        csv_writer.writerow(["tweet_date_created", "tweet_id", "tweet_text", "language", "sentiment", "sentiment_score"])
        with open(input_file, 'r', newline='') as csvinfile:
            csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                if row[4].upper() in ['POSITIVE', 'NEGATIVE']:
                    row_output = row
                    if str(row[4]) == 'POSITIVE':
                        row_output[4] = '1'
                    else:
                        row_output[4] = '0'
                    row_output = row
                    row_output[2] = clean_row(row_output[2])
                    #row_output[4] = one_hot(row[4],3)
                    #row_output = transform_instance(row_output)
                    #row_output[2] = nltk.word_tokenize(row_output[2])
                    csv_writer.writerow(row_output)# Preparing the training dataset

preprocess('betsentiment-ES-tweets-sentiment-teams.csv', 'data/tweets.train')


print('Loading data...')

#dataset = np.loadtxt("tweets.train",encoding='ISO-8859-1',delimiter=",")
#dataset2 = np.genfromtxt("tweets.train",encoding='ISO-8859-1',delimiter=",",dtype='object',usecols=2)
#dataset3 = np.genfromtxt("tweets.train",encoding='ISO-8859-1',delimiter=",",dtype='float',usecols=4)

#dataset2 = one_hot(dataset2, maxlen)
#print (str(dataset2),maxlen)
#for i in range(0,dataset2.size):
#    dataset2[i] = one_hot(dataset2[i], maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

datasetpanda = pd.read_csv("data/tweets.train", delimiter=",",encoding='ISO-8859-1')

datax = pd.DataFrame(datasetpanda,columns=['tweet_text'],dtype='str')
datay = pd.DataFrame(datasetpanda,columns=['sentiment'],dtype='str')

arrayX = []
for i in range(0,datax.size):
    listX = one_hot(str(datax.values[i]),maxlen)
    arrayX.append(listX)

arrayY = []
for i in range(0,datay.size):
    listY = datay.values[i]
    arrayY.append(listY)


#datasetpanda = pd.read_csv("tweets.train",encoding='ISO-8859-1',sep=',', header=None,engine='python')


#(x_train, y_train) = (datax, datay)
#(x_test, y_test) = (datax, datay)

arrayX = np.array(arrayX)
arrayY = np.array(arrayY)

(x_train, y_train) = (arrayX, arrayY)
(x_test, y_test) = (arrayX, arrayY)

#(x_train, y_train) = (dataset2, dataset2[:,1])
#(x_test, y_test) = (dataset2[:,0:8], dataset2[:,8])

#mlb = MultiLabelBinarizer()  # pass sparse_output=True if you'd like
#mlb.fit_transform(s.split(', ') for s in datax)



#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train,maxlen=maxlen,dtype='object')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen,dtype='object')


print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
