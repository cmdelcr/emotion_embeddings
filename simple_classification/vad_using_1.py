# Emotion classification using the approach of SAWE (Sentiment Aware Word Embeddings Using Refinement and Senti-Contextualized Learning Approach)
# using Bidirectional LSTM
# GLoVe 300
'''
acc:  0.8146286990508096
precision:  0.8871224165341812
recall:  0.8545176110260337
f1:  0.8705148205928237

acc:  0.8129536571747628
precision:  0.8726016884113584
recall:  0.8705972434915773
f1:  0.8715983135300882

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk import TweetTokenizer

import matplotlib.pyplot as plt


lstm_dim = 168
embedding_dim = 300
binary = True
epochs = 20
batch_size = 50


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
# reading vad_value
df = pd.read_csv('/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', sep='\t', header=None)

vad = {}
for index, row in df.iterrows():
    vad[row[0]] = [float(row[1]), float(row[2]), float(row[3])]


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


'''print('x_train: ', type(x_train), ', ', np.shape(x_train))
print('x_dev: ', type(x_dev), ', ', np.shape(x_dev))
print('x_test: ', type(x_test), ', ', np.shape(x_test))
print('y_train: ', type(y_train), ', ', np.shape(y_train))
print('y_dev: ', type(y_dev), ', ', np.shape(y_dev))
print('y_test: ', type(y_test), ', ', np.shape(y_test))
exit()
'''

# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
for line in open(os.path.join('../util/glove.6B.%sd.txt' % embedding_dim)):
	values = line.split()
	word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print("Number of word embeddings: ", len(word2vec))

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(word2idx) + 1
embedding_matrix = np.zeros((num_words, embedding_dim+3))
count_unk = 0
count_known = 0

for word, i in word2idx.items():
	embedding_vector = word2vec.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = np.append(embedding_vector, vad[word] if word in vad else np.zeros(3))
		count_known += 1
	else:
		embedding_matrix[i] = np.append(np.random.uniform(-0.25, 0.25, embedding_dim), vad[word] if word in vad else np.zeros(3))
		count_unk += 1

print('GLoVe loaded words: ', count_known)
print('Unknown words: ', count_unk)

pca = PCA(n_components=300)
embedding_matrix = pca.fit_transform(embedding_matrix)

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
