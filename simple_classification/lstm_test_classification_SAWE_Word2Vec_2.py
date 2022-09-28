# Emotion classification using the approach of SAWE (Sentiment Aware Word Embeddings Using Refinement and Senti-Contextualized Learning Approach)
# using Bidirectional LSTM
# Word2Vec
'''
acc:  0.8084868788386377
precision:  0.8824463860206513
recall:  0.8506891271056661
f1:  0.8662768031189084

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk import TweetTokenizer

from gensim.models import KeyedVectors

import matplotlib.pyplot as plt


lstm_dim = 168
embedding_dim = 300
binary = True
epochs = 20
batch_size = 100


def rem_urls(tokens):
	final = []
	for t in tokens:
		if t.startswith('@') or t.startswith('http') or t.find('www.') > -1 or t.find('.com') > -1:
			pass
		elif t[0].isdigit():
			final.append('NUMBER')
		else:
			final.append(t)
	return final

def read_datasets():
	datasets = {'train': [], 'dev': [], 'test': []}
	# TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
	tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
	for i in range(len(datasets)):
		for line in open(os.path.join('../../sota/SAWE-master/datasets/semeval', ('train' if i == 0 else 'dev' if i == 1 else 'test') + '.tsv')):
			idx, sidx, label, tweet = line.split('\t')
			if not (binary and ('neutral' in label or 'objective' in label)):
				datasets['train'].append((label, tweet)) if i == 0 else datasets['dev'].append((label, tweet)) if i == 1 else datasets['test'].append((label, tweet))

	y_train, x_train = zip(*datasets['train'])
	y_dev, x_dev = zip(*datasets['dev'])
	y_test, x_test = zip(*datasets['test'])


	x_train = [rem_urls(tknzr.tokenize(sent.lower())) for sent in x_train]
	y_train = np.asarray([[1, 0] if y == 'negative' else [0, 1] for y in y_train])

	x_dev = [rem_urls(tknzr.tokenize(sent.lower())) for sent in x_dev]
	y_dev = np.asarray([[1, 0] if y == 'negative' else [0, 1] for y in y_dev])

	x_test = [rem_urls(tknzr.tokenize(sent.lower())) for sent in x_test]
	y_test = np.asarray([[1, 0] if y == 'negative' else [0, 1] for y in y_test])


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



# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
# Load vectors directly from the file
model_word2vec = KeyedVectors.load_word2vec_format('/home/carolina/corpora/embeddings/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(word2idx) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
count_unk = 0
count_known = 0
for word, i in word2idx.items():
	try:
		embedding_vector = model_word2vec[word]
		embedding_matrix[i] = embedding_vector
		count_known += 1
	except:
		embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
		count_unk += 1

print('Word2vec loaded words: ', count_known)
print('Unknown words: ', count_unk)

embedding_layer = Embedding(
  num_words,
  embedding_dim,
  weights=[embedding_matrix],
  input_length=max_len_input,
  trainable=False
)


input_ = Input(shape=(max_len_input,))
x = embedding_layer(input_)
bidirectional = Bidirectional(LSTM(lstm_dim))
x1 = bidirectional(x)
output = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x1)


model = Model(inputs=input_, outputs=output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=batch_size, epochs=epochs, verbose=1)

pred = model.predict(x_test, verbose=1)
pred = np.where(pred > 0.5, 1, 0)

y_test = [np.argmax(y, axis=0) for y in y_test]
pred = [np.argmax(y, axis=0) for y in pred]

precision = precision_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
recall = recall_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
f1 = f1_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
acc = accuracy_score(y_true=y_test, y_pred=pred)

print('acc: ', acc)
print('precision: ', precision)
print('recall: ', recall)
print('f1: ', f1)
