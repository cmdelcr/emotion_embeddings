import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd
import settings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import preprocessing
import compress_files
import settings


stop_words = stopwords.words('english')
#max_num_words = 20000
embedding_dim = 300
lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
#lstm_dim_arr = [3]
#lstm_dim = 100

#lexicons = [settings.input_dir_lexicon_vad + '/home/carolina/corpora/lexicons/vad_lexicons/e-anew.csv', settings.input_dir_lexicon_vad + '/home/carolina/corpora/lexicons/vad_lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt']
#lexicons = [settings.input_dir_lexicon_vad + 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt']
#lexicons = [settings.input_dir_lexicon + 'NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt']
lemmatizer = WordNetLemmatizer()

#print(df.columns)
#dict_data = df.set_index('Word').T.to_dict(['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']) # does not work
arr_emo = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

dict_data_vad = {}
dict_data_emo_int = {}
inputs = []
y_train = []
mode = ['emo_int']


###VAD
'''print('Loading NRC-VAD ...')
df = pd.read_csv(settings.input_dir_lexicon_vad + 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', keep_default_na=False, header=None, sep='\t')
for index, row in df.iterrows(): #V, A, D
  val = len(str(row[0]).split())  
  dict_data_vad[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]
  inputs.append(str(row[0]).lower()) 
print('nrv_vad_size: ', len(dict_data_vad))'''



print('Loading NRC-Emo-Int ...')
df = pd.read_csv(settings.input_dir_lexicon + 'NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt', keep_default_na=False, header=None, sep='\t')
for index, row in df.iterrows():
  if str(row[0]) in dict_data_emo_int:
    arr_val = dict_data_emo_int[str(row[0])]
  else:
    arr_val = np.zeros((len(arr_emo)))
  arr_val[arr_emo.index(str(row[1]))] = float(row[2])
  dict_data_emo_int[str(row[0])] = arr_val
print('nrv_emo_int_size: ', len(dict_data_emo_int))

dict_voc = {}
data = []
idx = 0
inputs_keys = list(dict_data_emo_int.keys())
for key in inputs_keys:
  dict_voc[key] = idx
  idx += 1
  data.append(dict_data_emo_int[key])

data = preprocessing.normalize(data, norm='l1')

# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
for line in open(os.path.join(settings.input_dir_embeddings + '/glove/glove.6B.%sd.txt' % embedding_dim)):
  values = line.split()
  word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')


print('Addings lemmas')
counter_lem = 0
counter_word_dict = 0
counter_word = 0
arr_1 = {}
y_train = []
inputs = []
list_keys_w2v = list(word2vec.keys())
for key_w2v in list_keys_w2v:
  if key_w2v in dict_voc:
    inputs.append(key_w2v)
    counter_word_dict += 1
    arr_1[key_w2v] = counter_word
    counter_word += 1
    y_train.append(data[dict_voc[key_w2v]])
  else:
    lemma = lemmatizer.lemmatize(key_w2v)
    if lemma in dict_voc and lemma not in arr_1:
      counter_lem += 1
      inputs.append(key_w2v)
      arr_1[key_w2v] = counter_word
      counter_word += 1
      y_train.append(data[dict_voc[lemma]])

#print('words: ', counter_word_dict)
#print('lemmas: ', counter_lem)
#print("Number of total embeddings", len(arr_1))

for key in dict_voc.keys():
  if key not in arr_1:
    inputs.append(key)
    arr_1[key] = counter_word
    counter_word += 1
    y_train.append(data[dict_voc[key]])
#print(len(inputs))
#print(len(y_train))
#exit()

print("Number of word embeddings: ", len(word2vec))
print("Number of words in lexico", len(dict_data_emo_int))
print("Number of total embeddings", len(arr_1))
#exit()

y_train = np.asarray(y_train, dtype='float32')
minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
y_train = minmax_scale.fit_transform(y_train)

#print(stop_words)
# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(dict_voc)
embedding_matrix = np.zeros((len(arr_1), embedding_dim))
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
  dense = Dense(lstm_dim)
  x1 = dense(input_)
  output = Dense(len(arr_emo), activation='sigmoid')(x1)

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
  model.fit(embedding_matrix, y_train, batch_size=128, epochs=30, verbose=0)

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

  dir_name = '/home/carolina/embeddings/vad_emo-int'
  print(dir_name)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  name_file = os.path.join(dir_name, mode[0] + '_%d_lem.txt' % lstm_dim)
  with open(name_file, 'w') as f:
    i = 0
    mat = np.matrix(senti_embedding)
    for w_vec in mat:
      f.write(inputs[i].replace(" ", "_" ) + " ")
      np.savetxt(f, fmt='%.6f', X=w_vec)
      i += 1
    f.close()
  #compress_files.create_zip_file(name_file, name_file.replace('.txt', '.zip'))
  #os.remove(name_file)