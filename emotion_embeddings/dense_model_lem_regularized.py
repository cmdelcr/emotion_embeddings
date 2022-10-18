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
from sklearn import preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

#nltk.download('punkt')
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import settings

stop_words = stopwords.words('english')
#max_num_words = 20000
embedding_dim = 300
lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
#lstm_dim = 100

#lexicons = ['/home/carolina/corpora/lexicons/vad_lexicons/e-anew.csv', '/home/carolina/corpora/lexicons/vad_lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt']
lexicon = settings.input_dir_lexicon_vad + 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt'
lemmatizer = WordNetLemmatizer()

#print(df.columns)
#dict_data = df.set_index('Word').T.to_dict(['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']) # does not work

dict_data = {}
inputs = []
y_train = []
df = pd.read_csv(lexicon, keep_default_na=False, header=None, sep='\t')
max_len = 1
for index, row in df.iterrows(): #V, A, D
    val = len(str(row[0]).split())
    max_len = val if val > max_len else max_len
    dict_data[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]
    inputs.append(str(row[0]).lower()) 


print('Found %s unique input tokens.' % len(dict_data))

# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
for line in open(os.path.join(settings.input_dir_embeddings + 'glove/glove.6B.%sd.txt' % embedding_dim)):
  values = line.split()
  word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')
  #if str(values[0]) == 'soprano' or str(values[0]) == 'soprani':
  #  print(values[0])

print('Counting words')
counter_lem = 0
counter_word_dict = 0
counter_word = 0
arr_1 = {}
y_train = []
inputs = []
list_keys = list(word2vec.keys())
for key in list_keys:
  if key in dict_data:
    inputs.append(key)
    counter_word_dict += 1
    arr_1[key] = counter_word
    counter_word += 1
    y_train.append(dict_data[key])
  else:
    lemma = lemmatizer.lemmatize(key)
    if lemma in dict_data and lemma not in arr_1:
      counter_lem += 1
      inputs.append(key)
      arr_1[key] = counter_word
      counter_word += 1
      y_train.append(dict_data[lemma])

print("Number of word embeddings: ", len(word2vec))
print("Number of words in lexico", len(dict_data))
print("Number of total embeddings", counter_word_dict + counter_lem)


y_train = np.asarray(y_train, dtype='float32')
#minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#y_train = minmax_scale.fit_transform(y_train)

#print(stop_words)
# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(dict_data)
embedding_matrix = np.zeros((counter_word_dict + counter_lem, embedding_dim))
count_known_words = 0
count_unknown_words = 0
counter_stop_words = 0
for word, i in arr_1.items():
  #if i < max_num_words:
  embedding_vector = word2vec.get(word)
  if embedding_vector is None:
    # words not found in embedding index will be initialized with a gaussian distribution.
    embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    count_unknown_words += 1
  else:
    embedding_matrix[i] = embedding_vector
    count_known_words += 1

print('known_words: ', count_known_words)
print('unknown_words: ', count_unknown_words)
#exit()

print('Embedding matrix shape: ', np.shape(embedding_matrix))


for lstm_dim in lstm_dim_arr:
  input_ = Input(shape=(len(embedding_matrix[0]),))
  dense = Dense(lstm_dim, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activation="tanh")
  x1 = dense(input_)
  output = Dense(3, activation='linear')(x1)

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
  model.fit(embedding_matrix, y_train, batch_size=1024, epochs=200, verbose=1)

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

  dir_name = settings.local_dir_embeddings + 'dense_model_lem'
  if not os.path.exists(dir_name):
      os.makedirs(dir_name)
  with open(os.path.join(dir_name, 'emb_nrc_vad_lem_regularized%d.txt' % lstm_dim), 'w') as f:
      i = 0
      mat = np.matrix(senti_embedding)
      for w_vec in mat:
          f.write(inputs[i].replace(" ", "_" ) + " ")
          np.savetxt(f, fmt='%.6f', X=w_vec)
          i += 1
      f.close()