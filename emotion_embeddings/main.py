import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from gensim import models


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from DenseModel import DenseModel
from sklearn import preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from util import *
import settings


embedding_dim = 300
dim_arr = [10, 30, 50, 100, 200, 300]
arr_epochs = [50, 100, 200, 300, 400, 500]
arr_activation_functions = ['tanh', 'relu', 'sigmoid', 'exponential']
arr_type_matrix_emb = ['vad']


def create_model(input_shape, num_units, activation_function):
	input_ = Input(shape=(input_shape,))
	dense = Dense(num_units, activation=activation_function, 
		kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))
	x1 = dense(input_)
	output = Dense(3, activation='linear')(x1)

	model = Model(inputs=input_, outputs=output)

	return model

def compile_model(model, loss_='mean_squared_error', optimizer_='adam'):
	# regular categorical_crossentropy requires one_hot_encoding for the targets, 
	#sparse_categorical_crossentropy is used to don't use the conversion
	model.compile(
			loss=loss_,
			optimizer=optimizer_,#Adam(lr=0.001),
			metrics=['accuracy']
	)

	return model

def train_model(model, x_train, y_train, batch_size_=1024, epochs_=200, verbose=0):
	print('Training model...')
	r = model.fit(x_train, 
		y_train, 
		batch_size=batch_size_, 
		epochs=epochs_, 
		verbose=0)

	return r

def evaluate_model(model, y_train, plot_results=False):
	results = model.evaluate(embedding_matrix, y_train)
	print("train loss, train acc:", results)

	if plot_results:
		plt.plot(r.history['loss'], label='loss')
		plt.legend()
		plt.show()

		# accuracies
		plt.plot(r.history['accuracy'], label='acc')
		plt.legend()
		plt.show()


def merge_semantic_end_emotion_embeddings(model, embedding_matrix, act='tanh', apply_pca=True, print_sizes=False):
	if print_sizes:
		print('Matrix input_to_dense: ', np.shape(model.layers[1].get_weights()[0]))
		print('Bias input_to_dense: ', np.shape(model.layers[1].get_weights()[1]))
		print('Matrix dense_to_output: ', np.shape(model.layers[2].get_weights()[0]))
		print('Bias dense_to_output', np.shape(model.layers[2].get_weights()[1]))

	input_matrix_dense = model.layers[1].get_weights()[0]
	input_bias_dense = model.layers[1].get_weights()[1]
	output_matrix_dense = model.layers[2].get_weights()[0]
	output_bias_dense = model.layers[2].get_weights()[1]

	senti_embedding = embedding_matrix
	#print('^^^^')
	#print(np.shape(senti_embedding))
	senti_embedding = np.dot(embedding_matrix, input_matrix_dense) + input_bias_dense
	#print(np.shape(senti_embedding))
	senti_embedding = np.apply_along_axis(np.tanh, 0, senti_embedding)
	#print(np.shape(senti_embedding))
	senti_embedding_no_pca = np.hstack((embedding_matrix, senti_embedding))
	#print(np.shape(senti_embedding))
	#exit()

	#if apply_pca:
	pca = PCA(n_components=300)
	senti_embedding = pca.fit_transform(senti_embedding_no_pca)

	return senti_embedding, senti_embedding_no_pca



#------------------------------------------------------------------------------------------------------------------
dict_data = read_vad_file()
print('NRC-VAD size: ', len(dict_data))

activation_function = 'tanh'
for emb_type in settings.embedding_type:
	word2vec = read_embeddings(emb_type)
	word2idx, vocabulary, y_train = getting_lemmas(emb_type, dict_data, word2vec)
	
	for type_matrix_emb in arr_type_matrix_emb:
		print('||||||||    Type matrix emp: ', type_matrix_emb)
		if emb_type == 'word2vec' and type_matrix_emb == 'full':
			continue
		embedding_matrix, vocabulary_, y_train_ = filling_embeddings(word2idx, word2vec, vocabulary, embedding_dim, emb_type, type_matrix_emb, y_train)
		print('Embeddings matrix shape: ', embedding_matrix.shape)

		for embedding_dimention in dim_arr:
			print('--------------------------')
			print('\nFor hidden size: ',  embedding_dimention)
			print('Creating the model...')
			for act in arr_activation_functions:
				print('##########   Activation: ', act)
				for epoch in arr_epochs:
					print('$$$$  Epochs: ', epoch)
					#model = DenseModel(embedding_dimention, act)
					model = create_model(len(embedding_matrix[0]), embedding_dimention, act)
					model = compile_model(model)
					r = train_model(model, embedding_matrix, y_train_, epochs_=epoch)
					evaluate_model(model, y_train_)
					senti_embedding, senti_embedding_ = merge_semantic_end_emotion_embeddings(model, embedding_matrix, act)
					print('Senti embeddings size PCA', np.shape(senti_embedding))
					print('Senti embeddings size no PCA', np.shape(senti_embedding_))
					print('----------------------------------------')

					name_file = 'sent_emb_' + emb_type + '_' + str(embedding_dimention) + '_' + act + '_e'+ str(epoch) + '_pca_' + type_matrix_emb + '.txt'
					save_senti_embeddings(senti_embedding, vocabulary, vocabulary_, name_file)
					name_file = 'sent_emb_' + emb_type + '_' + str(embedding_dimention) + '_' + act + '_e'+ str(epoch) + '_nopca_' + type_matrix_emb + '.txt'
					save_senti_embeddings(senti_embedding_, vocabulary, vocabulary_, name_file)
				print('..............................')


	print('------------------------------------------------------------------------------------------------------')
	print('------------------------------------------------------------------------------------------------------\n')