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
    dict_data[str(row[0]).lower()] = np.asarray([float(row[1]), float(row[2]), float(row[3])])
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

len_words_not_found = 0
for word in dict_data.keys():
  if word not in word2vec:
    len_words_not_found += 1
print('len_words_not_found: ', len_words_not_found)

y_train = np.asarray(y_train, dtype='float32')
#minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#y_train = minmax_scale.fit_transform(y_train)

#print(stop_words)
# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(dict_data)
embedding_matrix = np.zeros((len(word2vec) + len_words_not_found, embedding_dim+3))
count_known_words = 0
count_unknown_words = 0
counter_stop_words = 0
i = 0
last_keys = []
inputs_all = []
for key in word2vec.keys():
  if key in arr_1:
    embedding_matrix[i][0:300] = word2vec.get(key)
    embedding_matrix[i][300:303] = y_train[arr_1[key]]
    last_keys.append(key)
  else:
    embedding_matrix[i][0:300] = word2vec.get(key)
    embedding_matrix[i][300:303] = np.asarray([0.5, 0.5, 0.5])
  i += 1
  inputs_all.append(key)

for word in dict_data.keys():
  if word not in last_keys:
    embedding_matrix[i][0:300] = np.random.uniform(-0.25, 0.25, embedding_dim)
    embedding_matrix[i][300:303] = dict_data[word]
    inputs_all.append(word)
    i += 1

embedding_matrix = np.asarray(embedding_matrix)
print(np.shape(embedding_matrix))


print('Embedding matrix shape: ', np.shape(embedding_matrix))



pca = PCA(n_components=300)
senti_embedding = pca.fit_transform(embedding_matrix)


print(np.shape(senti_embedding))

dir_name = settings.local_dir_embeddings + 'concatenate_vad'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(os.path.join(dir_name, 'concatenate_vad_%d.txt' % 300), 'w') as f:
    i = 0
    mat = np.matrix(senti_embedding)
    for w_vec in mat:
        f.write(inputs_all[i].replace(" ", "_" ) + " ")
        np.savetxt(f, fmt='%.6f', X=w_vec)
        i += 1
    f.close()