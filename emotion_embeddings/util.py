import os
from gensim import models
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
import settings


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

def read_vad_file():
	dict_data = {}
	df = pd.read_csv(settings.input_dir_lexicon_vad + 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', 
				keep_default_na=False, header=None, sep='\t')
	for index, row in df.iterrows(): #V, A, D
		dict_data[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]

	return dict_data

def getting_lemmas(emb_type, vad, word2vec):
	counter_lem = 0
	counter_word_dict = 0
	counter_word = 0
	word2idx = {}
	vad_value = []
	vocabulary = []

	lemmatizer = WordNetLemmatizer()
	list_keys = list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys())

	for key in list_keys:
		if key in vad:
			vocabulary.append(key)
			counter_word_dict += 1
			word2idx[key] = counter_word
			counter_word += 1
			vad_value.append(vad[key])
		else:
			lemma = lemmatizer.lemmatize(key)
			if lemma in vad and lemma not in word2idx:
				counter_lem += 1
				vocabulary.append(key)
				word2idx[key] = counter_word
				counter_word += 1
				vad_value.append(vad[lemma])

	print('words in vad and ' + emb_type + ': ', counter_word_dict)
	print('lemmas: ', counter_lem)
	print('final vocabulary size: ', counter_word)

	return word2idx, vocabulary, vad_value

def filling_embeddings(word2idx, word2vec, vocabulary, embedding_dim, emb_type, type_matrix_emb, y_train):
	print('***Filling pre-trained embeddings...')
	count_known_words = 0
	count_unknown_words = 0
	counter_stop_words = 0

	if type_matrix_emb == 'vad': 
		embedding_matrix = np.zeros((len(vocabulary), embedding_dim))
		for word, i in word2idx.items():
			embedding_vector = word2vec[word]
			if embedding_vector is None:
				# words not found in embedding index will be initialized with a gaussian distribution.
				embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
				count_unknown_words += 1
			else:
				embedding_matrix[i] = embedding_vector
				count_known_words += 1
		y_train = np.asarray(y_train, dtype='float32')
	else:
		set1 = set(vocabulary)
		vocabulary_word2vec = list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys())
		set2 = set(vocabulary_word2vec)
		set_all = set.union(set1, set2)
		embedding_matrix = np.zeros((len(set_all), embedding_dim))
		i = 0
		y_train_ = []
		vocabulary_ = []
		for word in vocabulary_word2vec:
			embedding_matrix[i] = word2vec[word]
			if word in vocabulary:
				y_train_.append(y_train[vocabulary.index(word)])
			else:
				y_train_.append([0.5, 0.5, 0.5])
			vocabulary_.append(word)
			i += 1
		keys_rest = list(set1 - set2)
		for word in keys_rest:
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
			vocabulary_.append(word)
			y_train_.append(y_train[vocabulary.index(word)])
			i += 1
		vocabulary = vocabulary_
		y_train = np.asarray(y_train_, dtype='float32')

	
	return embedding_matrix, vocabulary, y_train

def save_senti_embeddings(senti_embedding, labels, labels_, name_file):
	dir_name = settings.local_dir_embeddings + 'dense_model'
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	with open(os.path.join(dir_name, name_file), 'w') as f:
		i = 0
		mat = np.matrix(senti_embedding)
		for w_vec in mat:
			if labels_[i] in labels:
				f.write(labels_[i].replace(" ", "_" ) + " ")
				np.savetxt(f, fmt='%.6f', X=w_vec)
			i += 1
		f.close()


