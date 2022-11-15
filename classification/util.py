import re
import os
import pandas as pd
from string import punctuation
from nltk import TweetTokenizer

from gensim import models
import settings

import numpy as np

emotions = ['negative', 'positive']
punctuation_list = list(punctuation)

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

def read_datasets(dir_datasets):
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

	#y_train = np.asarray([[1, 0] if val == 0 else [0, 1] for val in y_train])
	#y_dev = np.asarray([[1, 0] if val == 0 else [0, 1] for val in y_dev])
	#y_test = np.asarray([[1, 0] if val == 0 else [0, 1] for val in y_test])
	y_train = np.asarray(y_train)
	y_dev = np.asarray(y_dev)
	y_test = np.asarray(y_test)


	return y_train, x_train, y_dev, x_dev, y_test, x_test

def read_embeddings(type_emb='glove'):
	print('\n-----------------------------------------')
	print('Loading embeddings ' + type_emb + '...')
	print('-----------------------------------------')

	word2vec = {}
	if type_emb == 'word2vec':
		word2vec = models.KeyedVectors.load_word2vec_format(settings.dir_embeddings_word2vec, binary=True)
		#word2vec.index_to_key
	else:
		path = settings.dir_embeddings_glove if type_emb == 'glove' else settings.dir_embeddings_numberbatch
		for line in open(path):
			values = line.split()
			word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')

	return word2vec		


def read_embeddings_generated(str_dir):
	print('\n-----------------------------------------')
	print('Loading embeddings ' + re.sub(r'\.txt', '', str_dir) + '...')

	word2vec = {}
	path = settings.local_dir_embeddings + 'dense_model/' + str_dir
	for line in open(path):
		values = line.split()
		word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')

	return word2vec		