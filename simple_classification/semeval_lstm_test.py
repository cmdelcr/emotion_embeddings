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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn import preprocessing
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

from nltk import TweetTokenizer

import matplotlib.pyplot as plt



lstm_dim = 168
embedding_dim = 300
binary = True
epochs = 20
batch_size = 25
#lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
lstm_dim_arr = [300]
#lexicons = ['e-anew', 'nrc_vad']
lexicons = ['nrc_vad']


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
	y_train = np.asarray([0 if y == 'negative' else 1 for y in y_train])

	x_dev = [rem_urls(tknzr.tokenize(sent.lower())) for sent in x_dev]
	y_dev = np.asarray([0 if y == 'negative' else 1 for y in y_dev])

	x_test = [rem_urls(tknzr.tokenize(sent.lower())) for sent in x_test]
	y_test = np.asarray([0 if y == 'negative' else 1 for y in y_test])


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

#####################################################################################################################3
'''df = pd.read_csv('/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', keep_default_na=False, header=None, sep='\t')
max_len = 1
dict_data = {}
for index, row in df.iterrows(): #V, A, D
  val = len(str(row[0]).split())
  max_len = val if val > max_len else max_len
  dict_data[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]'''
#####################################################################################################################3


for lexico in lexicons:
	for lstm_dim_vec in lstm_dim_arr:
		# store all the pre-trained word vectors
		print('Loading word vectors...')
		word2vec = {}
		lexico = 'nrc_vad'
		lstm_dim_vec = 300
		for line in open('../emotion_embeddings/embeddings/senti-embedding/emb_' + lexico + '_%ddim_scaled_extended.txt' % 100):
		#for line in open(os.path.join('../util/glove.6B.%sd.txt' % embedding_dim)):
		#for line in open('../util/ewe_uni.txt'): # 183712
		#for line in open('../util/sawe-tanh-pca-100-glove.txt'): # 30958 ,13916-e-anew
			values = line.split()
			word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
		print("Number of word embeddings: ", len(word2vec))
		#word2vec = KeyedVectors.load_word2vec_format('/home/carolina/corpora/embeddings/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
		#exit()
		count_missing_words = 0
		# prepare embedding matrix
		print('Filling pre-trained embeddings...')
		num_words = len(word2idx) + 1
		embedding_matrix = np.zeros((num_words, embedding_dim))
		for word, i in word2idx.items():
			embedding_vector = word2vec.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all zeros.
				embedding_matrix[i] = embedding_vector
				#if word not in dict_data:
				#	count_missing_words += 1
			else:
				embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)

		#print(count_missing_words)
		#exit()
		'''count_known = 0
		count_unk = 0
		for word, i in word2idx.items():
			try:
				embedding_vector = word2vec[word]
				embedding_matrix[i] = embedding_vector
				count_known += 1
			except:
				embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
				count_unk += 1

		print('Word2vec loaded words: ', count_known)
		print('Unknown words: ', count_unk)'''
		
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
		output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x1)


		model = Model(inputs=input_, outputs=output)
		model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
		model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=batch_size, epochs=epochs, verbose=1)

		pred = model.predict(x_test, verbose=1)
		pred = np.where(pred > 0.5, 1, 0)

		precision = precision_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
		recall = recall_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
		f1 = f1_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
		acc = accuracy_score(y_true=y_test, y_pred=pred)
		r2 = r2_score(y_true=y_test, y_pred=pred)


		#print('Lexico: ', lexico)
		#print('Emo_emb_size: ', lstm_dim_vec)
		print('acc: ', acc)
		print('precision: ', precision)
		print('recall: ', recall)
		print('f1: ', f1)
		print('r2: ', r2)
		print('------------------------------------------')

		#with open('results.csv', 'a') as file:
		#	file.write(',' + lexico + ',' + str(lstm_dim_vec) + ',' + str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')
		#	file.close()
