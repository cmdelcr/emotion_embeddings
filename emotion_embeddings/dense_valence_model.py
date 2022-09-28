# Simple emotion classification considering the isear dataset with 7 emotions (joy, fear, anger, sadness, disgust, shame, and guilt)
# using Bidirectional LSTM, dropout and a dense layer1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

nltk.download('wordnet')


#max_num_words = 20000
embedding_dim = 300
lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
#lstm_dim = 100

#lexicons = ['/home/carolina/corpora/lexicons/e-anew.csv', '/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt']
lexicons = ['/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt']

lemmatizer = WordNetLemmatizer()

for lexico in lexicons:
  #print(df.columns)
  #dict_data = df.set_index('Word').T.to_dict(['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']) # does not work

  dict_data = {}
  if 'e-anew' in lexico:
    df = pd.read_csv(lexico, keep_default_na=False)
    for index, row in df.iterrows():
      vad_arr = [float(row['V.Mean.Sum']), float(row['A.Mean.Sum']), float(row['D.Mean.Sum'])]
      dict_data[str(row['Word'])] = vad_arr
  else:
    df = pd.read_csv(lexico, keep_default_na=False, header=None, sep='\t')
    max_len = 1
    inx = 0
    for index, row in df.iterrows(): #V, A, D
      word = str(row[0])
      '''if word != lemmatizer.lemmatize(word):
        print(word, ', ', lemmatizer.lemmatize(word))
        inx += 1
      if inx > 10:
        exit()'''
      val = len(str(row[0]).split())
      max_len = val if val > max_len else max_len
      dict_data[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]

  print(max_len)
  inputs = list(dict_data.keys())
  tokenizer = Tokenizer()#num_words=max_num_words)
  tokenizer.fit_on_texts(inputs)
  x_train = tokenizer.texts_to_sequences(inputs)
  #x_train = np.asarray(x_train)

  # get the word to index mapping for input language
  word2idx = tokenizer.word_index
  print('Found %s unique input tokens.' % len(word2idx))

  y_train = []
  for val in inputs:
    y_train.append(dict_data[val])

  y_train = np.asarray(y_train, dtype='float32')

  # store all the pre-trained word vectors
  print('Loading word vectors...')
  word2vec = {}
  for line in open(os.path.join('../util/glove.6B.%sd.txt' % embedding_dim)):
  	values = line.split()
  	word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
  print("Number of word embeddings: ", len(word2vec))


  # prepare embedding matrix
  print('Filling pre-trained embeddings...')
  num_words = len(inputs)
  embedding_matrix = np.zeros((num_words, embedding_dim))
  for word, i in word2idx.items():
    #if i < max_num_words:
    embedding_vector = word2vec.get(word)
    if embedding_vector is None:
      # words not found in embedding index will be initialized with a gaussian distribution.
      embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    else:
      embedding_matrix[i] = embedding_vector

  if 'e-anew' in lexico:
    embedding_matrix = embedding_matrix / 10

  print('Embedding matrix shape: ', np.shape(embedding_matrix))

 
  for lstm_dim in lstm_dim_arr:
    input_ = Input(shape=(len(embedding_matrix[0]),))
    dense = Dense(lstm_dim)
    x1 = dense(input_)
    output = Dense(3, activation='sigmoid')(x1)

    model = Model(inputs=input_, outputs=output)

    # compile
    model.compile(
      # regular categorical_crossentropy requires one_hot_encoding for the targets, sparse_categorical_crossentropy is used to don't use the conversion
      loss='mean_absolute_error',
      optimizer='adam',#Adam(lr=0.001),
      metrics=['accuracy']
    )

    # train
    print('Training model...')
    model.fit(embedding_matrix, y_train, batch_size=128, epochs=200, verbose=0)

    print('Matrix input_to_dense: ', np.shape(model.layers[1].get_weights()[0]))
    print('Bias input_to_dense: ', np.shape(model.layers[1].get_weights()[1]))
    print('Matrix dense_to_output: ', np.shape(model.layers[2].get_weights()[0]))
    print('Bias dense_to_output', np.shape(model.layers[2].get_weights()[1]))

    input_matrix_dense = model.layers[1].get_weights()[0]
    input_bias_dense = model.layers[1].get_weights()[1]
    output_matrix_dense = model.layers[2].get_weights()[0]
    output_bias_dense = model.layers[2].get_weights()[1]

    senti_embedding = embedding_matrix
    senti_embedding = np.dot(embedding_matrix, input_matrix_dense) + input_bias_dense
    senti_embedding = np.apply_along_axis(np.tanh, 0, senti_embedding)
    senti_embedding = np.hstack((embedding_matrix, senti_embedding))
    pca = PCA(n_components=300)
    senti_embedding = pca.fit_transform(senti_embedding)


    print(np.shape(senti_embedding))

    dir_name = 'embeddings/senti-embedding-modif/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, 'emb_' + ('e-anew' if 'e-anew' in lexico else 'nrc_vad') + '_%ddim_2.txt' % lstm_dim), 'w') as f:
        i = 0
        mat = np.matrix(senti_embedding)
        for w_vec in mat:
            f.write(inputs[i].replace(" ", "_" ) + " ")
            np.savetxt(f, fmt='%.6f', X=w_vec)
            i += 1
        f.close()