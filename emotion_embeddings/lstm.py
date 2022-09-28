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

import matplotlib.pyplot as plt

#nltk.download('punkt')


#max_num_words = 20000
embedding_dim = 300
lstm_dim = 100


df = pd.read_csv('/home/carolina/corpora/lexicons/e-anew.csv', keep_default_na=False)
#print(df.columns)
#dict_data = df.set_index('Word').T.to_dict(['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']) # does not work
dict_data = {}
for index, row in df.iterrows():
    dict_data[str(row['Word'])] = [float(row['V.Mean.Sum']), float(row['A.Mean.Sum']), float(row['D.Mean.Sum'])]

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

# determine maximum length input sequence
#max_len_input = max(len(s) for s in input_sequences)

#inputs = pad_sequences(input_sequences, maxlen=max_len_input)
# when padding is not specified it takes the default at the begining of the sentence
#print("inputs.shape:", inputs.shape)
#print("inputs[0]:", inputs[0])


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
  if embedding_vector is not None:
    # words not found in embedding index will be initialized with a gaussian distribution.
    embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)

embedding_matrix = embedding_matrix / 10

print('Embedding matrix shape: ', np.shape(embedding_matrix))

# Perform one-hot encoding on df[0] i.e emotion
#enc = OneHotEncoder()#handle_unknown='ignore')
#outputs = enc.fit_transform(np.array(df[0]).reshape(-1,1)).toarray()


# Split into train and test
#X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)



# lstm_dim=168
# create embedding layer
'''embedding_layer = Embedding(
  num_words,
  embedding_dim,
  weights=[embedding_matrix],
  #input_length=max_len_input,
  trainable=False
)'''

input_ = Input(shape=(len(embedding_matrix[0]),))
#x = embedding_layer(input_)
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
model.fit(embedding_matrix, y_train, batch_size=128, epochs=200)


#for layer in model.layers:
#  print(layer.name, layer)
#  print(np.shape(layer.get_weights()))
#  print('-------------------------------------------------------')

print('Matrix input_to_dense: ', np.shape(model.layers[1].get_weights()[0]))
print('Bias input_to_dense: ', np.shape(model.layers[1].get_weights()[1]))
print('Matrix dense_to_output: ', np.shape(model.layers[2].get_weights()[0]))
print('Bias dense_to_output', np.shape(model.layers[2].get_weights()[1]))

input_matrix_dense = model.layers[1].get_weights()[0]
input_bias_dense = model.layers[1].get_weights()[1]
output_matrix_dense = model.layers[2].get_weights()[0]
output_bias_dense = model.layers[2].get_weights()[1]

senti_embedding = embedding_matrix
#senti_embedding = np.dot(embedding_matrix, input_matrix_dense) + input_bias_dense
#senti_embedding = np.apply_along_axis(np.tanh, 0, senti_embedding)
#senti_embedding = np.hstack((embedding_matrix, senti_embedding))
#pca = PCA(n_components=300)
#senti_embedding = pca.fit_transform(senti_embedding)


print(np.shape(senti_embedding))

dir_name = 'embeddings/senti-embedding/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(os.path.join(dir_name, 'emb_100dim.txt'), 'w') as f:
    i = 0
    mat = np.matrix(senti_embedding)
    for w_vec in mat:
        f.write(inputs[i].replace(" ", "_" ) + " ")
        np.savetxt(f, fmt='%.6f', X=w_vec)
        i += 1
    f.close()