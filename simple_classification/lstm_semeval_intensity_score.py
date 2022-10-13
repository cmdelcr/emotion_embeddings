# Emotion classification using the approach of SAWE (Sentiment Aware Word Embeddings Using Refinement and Senti-Contextualized Learning Approach)
# using Bidirectional LSTM
# GLoVe 300
'''
acc:  0.8135120044667783
precision:  0.8790951638065523
recall:  0.8629402756508423
f1:  0.8709428129829986
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.optimizers import SGD, Adam

import pandas as pd
import re

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, GRU, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors
from sklearn import preprocessing

from gensim.models import KeyedVectors

from nltk import TweetTokenizer

import matplotlib.pyplot as plt
import settings
from string import punctuation

punctuation_list = list(punctuation)

lstm_dim = 200
embedding_dim = 300
binary = True
epochs = 30
batch_size = 25
#lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
lstm_dim_arr = [300]
#lexicons = ['e-anew', 'nrc_vad']
#lexicons = ['nrc_vad']
mode = ['vad_lem']#vad_emo-int
dir_datasets = settings.input_dir_emo_corpora + 'semeval/semeval_2018/English/EI-reg/'
emotions = ['anger', 'fear', 'joy', 'sadness']


def remove_unecesary_data(sent):
	# remove urls (https?:\/\/\S+) --> for urls with http
	sent = re.sub(r'https?:\/\/\S+', '', sent)
	sent = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', sent)
	# remove html reference characters
	sent = re.sub(r'&[a-z]+;', '', sent)
	#remove non-letter characters
	sent = re.sub(r"[a-z\s\(\-:\)\\\/\\];='#", "", sent)
	#removing handles
	sent = re.sub(r'@[a-zA-Z0-9-_]*', '', sent)
	# remove the symbol from hastag to analize the word
	sent = re.sub(r'#', '', sent)

	return sent


def preprocessing(sent):
	sent = remove_unecesary_data(sent)

	tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
	tokens = tknzr.tokenize(sent)

	return [w for w in tokens if w not in punctuation_list]


def read_datasets():
	datasets = {'train': [], 'dev': [], 'test': []}
	# TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
	for file in os.listdir(dir_datasets):
		key = re.sub('.txt', '', file)
		df = pd.read_csv(dir_datasets + file, sep='\t')
		for index, row in df.iterrows():
			arr = np.zeros((4))
			arr[emotions.index(str(row[2]))] = float(row[3])
			datasets[key].append((arr, str(row[1])))

	y_train, x_train = zip(*datasets['train'])
	y_dev, x_dev = zip(*datasets['dev'])
	y_test, x_test = zip(*datasets['test'])

	x_train = [preprocessing(sent.lower()) for sent in x_train]
	x_dev = [preprocessing(sent.lower()) for sent in x_dev]
	x_test = [preprocessing(sent.lower()) for sent in x_test]

	y_train = np.asarray(y_train)
	y_dev = np.asarray(y_dev)
	y_test = np.asarray(y_test)


	return y_train, x_train, y_dev, x_dev, y_test, x_test




y_train, x_train, y_dev, x_dev, y_test, x_test = read_datasets()


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train + x_dev + x_test)
x_train = tokenizer.texts_to_sequences(x_train)
x_dev = tokenizer.texts_to_sequences(x_dev)
x_test = tokenizer.texts_to_sequences(x_test)

# get the word to index mapping for input language
word2idx = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx))

# determine maximum length input sequence
max_len_input = max(len(s) for s in x_train + x_dev + x_test)

x_train = pad_sequences(x_train, max_len_input, padding='pre', truncating='post')
x_dev = pad_sequences(x_dev, max_len_input, padding='pre', truncating='post')
x_test = pad_sequences(x_test, max_len_input, padding='pre', truncating='post')


for lstm_dim_vec in lstm_dim_arr:
	# store all the pre-trained word vectors
	print('Loading word vectors...')
	word2vec = {}
	lexico = 'nrc_vad'
	for line in open(settings.input_dir_embeddings + 'glove/glove.6B.%sd.txt' % embedding_dim):
		values = line.split()
		word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
	print("Number of word embeddings: ", len(word2vec))

	count_missing_words = 0
	# prepare embedding matrix
	print('Filling pre-trained embeddings...')
	num_words = len(word2idx) + 1
	embedding_matrix = np.zeros((num_words, embedding_dim))
	for word, i in word2idx.items():
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)

	embedding_layer = Embedding(
	  num_words,
	  embedding_dim,
	  weights=[embedding_matrix],
	  input_length=max_len_input,
	  trainable=False
	)


	input_ = Input(shape=(max_len_input,))
	x = embedding_layer(input_)
	bidirectional = (GRU(250))
	x1 = bidirectional(x)
	x1 = Dense(100, activation='relu')(x1)
	output = Dense(4, activation='linear')(x1)


	model = Model(inputs=input_, outputs=output)
	#opt = SGD(learning_rate=0.01, momentum=0.9)
	opt = Adam(learning_rate=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=batch_size, epochs=epochs, verbose=1)

	pred = model.predict(x_test, verbose=1)

	r2 = r2_score(y_true=y_test, y_pred=pred)
	#accuracy = accuracy_score(y_true=y_test, y_pred=pred)


	#print('Lexico: ', lexico)
	print('Emo_emb_size: ', lstm_dim_vec)
	#print('accuracy: ', accuracy)
	print('r2: ', r2)
	print('------------------------------------------')

	#with open('results.csv', 'a') as file:
	#	file.write(',' + lexico + ',' + str(lstm_dim_vec) + ',' + str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')
	#	file.close()
