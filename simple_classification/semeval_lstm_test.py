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
import seaborn as sn
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

from nltk import TweetTokenizer

import matplotlib.pyplot as plt
import settings
import re
from string import punctuation

from collections import Counter
import statistics


punctuation_list = list(punctuation)


lstm_dim = 168
embedding_dim = 300
binary = True
epochs = 20
batch_size = 25
lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
#lstm_dim_arr = [300]
mode = ['vad_lem']#vad_emo-int
dir_datasets = settings.input_dir_emo_corpora + 'semeval/semeval_2013/'
emotions = ['negative', 'positive']


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
	
	for file in os.listdir(dir_datasets):
		key = re.sub('.tsv', '', file)
		df = pd.read_csv(dir_datasets + file, sep='\t')

		for index, row in df.iterrows():
			if str(row[2]) in emotions:
				datasets[key].append((emotions.index(str(row[2])), str(row[3])))
	
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

#train: {0: 1159, 1: 2973}
#dev:   {0: 280,  1: 483}
#test:  {0: 472,  1: 1280}

#train = dict(Counter(y_train))
#print(train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train + x_dev)
x_train = tokenizer.texts_to_sequences(x_train)
x_dev = tokenizer.texts_to_sequences(x_dev)
x_test = tokenizer.texts_to_sequences(x_test)

# get the word to index mapping for input language
word2idx = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx))

# determine maximum length input sequence
max_len_input = max(len(s) for s in x_train + x_dev)

x_train = pad_sequences(x_train, max_len_input, padding='pre', truncating='post')
x_dev = pad_sequences(x_dev, max_len_input, padding='pre', truncating='post')
x_test = pad_sequences(x_test, max_len_input, padding='pre', truncating='post')

act = 'lem'
#for lexico in lexicons:
for lstm_dim_vec in lstm_dim_arr:
	# store all the pre-trained word vectors
	print('Loading word vectors...')
	word2vec = {}
	lexico = 'nrc_vad'
	#lstm_dim_vec = 300
	#for line in open(settings.local_dir_embeddings + 'sota/mewe_embeddings/emo_embeddings.txt'):
	#for line in open(settings.local_dir_embeddings + mode[0] + '/emo_int_%d_lem.txt' % lstm_dim_vec):
	#for line in open(settings.local_dir_embeddings + mode[0] + '/vad_lem_%d.txt' % lstm_dim_vec):
	for line in open(settings.local_dir_embeddings + 'dense_model_%s/emb_nrc_vad_lem_%d.txt' % (act, lstm_dim_vec)):
	#for line in open(settings.input_dir_embeddings + 'glove/glove.6B.%sd.txt' % embedding_dim):
	#for line in open(settings.input_dir_senti_embeddings + 'ewe_uni.txt'):
	#for line in open(settings.input_dir_senti_embeddings + 'sawe-tanh-pca-100-glove.txt'):
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

	arr_acc = []
	arr_precision = []
	arr_recall = []
	arr_f1 = []

	model = Model(inputs=input_, outputs=output)
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=batch_size, epochs=epochs, verbose=0)

	pred = model.predict(x_test, verbose=1)
	pred = np.where(pred > 0.5, 1, 0)

	precision = precision_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
	recall = recall_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
	f1 = f1_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
	acc = accuracy_score(y_true=y_test, y_pred=pred)
	r2 = r2_score(y_true=y_test, y_pred=pred)


	#print('Lexico: ', lexico)
	print('Emo_emb_size: ', lstm_dim_vec)
	print('acc: ', acc)
	print('precision: ', precision)
	print('recall: ', recall)
	print('f1: ', f1)
	print('------------------------------------------')
	#arr_acc.append(acc)
	#arr_precision.append(precision)
	#arr_recall.append(recall)
	#arr_f1.append(f1)


	dir_name = '../results/dense_model/'
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	with open(dir_name + 'results.csv', 'a') as file:
		file.write('glove\tnrc_vad_' + act + '\t' + str(lstm_dim_vec) + '\t%.6f\t%.6f\t%.6f\t%.6f\n' % (acc, precision, recall, f1))
		file.close()
	#embeddings	lexico	size_emo_emb	accuracy	precision	recall	f1_score
	'''with open('../results/results_binary_classification_semeva13.csv', 'a') as file:
		file.write('glove\t\t' + str(embedding_dim) + '\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\n' %
		 (statistics.mean(arr_acc), statistics.pstdev(arr_acc), statistics.mean(arr_precision), statistics.pstdev(arr_precision),
		 	statistics.mean(arr_recall), statistics.pstdev(arr_recall), statistics.mean(arr_f1), statistics.pstdev(arr_f1)))
		file.close()'''
