# Simple emotion classification considering the isear dataset with 7 emotions (joy, fear, anger, sadness, disgust, shame, and guilt)
# using Bidirectional LSTM, dropout and a dense layer1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

#nltk.download('punkt')


max_num_words = 20000
embedding_dim = 50
lstm_dim = 100


df = pd.read_csv('../util/isear.csv',header=None)
# Remove 'No response' row value in isear.csv
df = df[~df[1].str.contains("NO RESPONSE")]
#print(df[0].unique())
#exit()

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(df[1])
input_sequences = tokenizer.texts_to_sequences(df[1])

# get the word to index mapping for input language
word2idx = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

inputs = pad_sequences(input_sequences, maxlen=max_len_input)
# when padding is not specified it takes the default at the begining of the sentence
#print("inputs.shape:", inputs.shape)
print("inputs[0]:", inputs[0])


# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
for line in open(os.path.join('../util/glove.6B.%sd.txt' % embedding_dim)):
	values = line.split()
	word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print("Number of word embeddings: ", len(word2vec))


# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(max_num_words, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word2idx.items():
  if i < max_num_words:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector



# Perform one-hot encoding on df[0] i.e emotion
enc = OneHotEncoder()#handle_unknown='ignore')
outputs = enc.fit_transform(np.array(df[0]).reshape(-1,1)).toarray()


# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)


# create embedding layer
embedding_layer = Embedding(
  num_words,
  embedding_dim,
  weights=[embedding_matrix],
  input_length=max_len_input,
  # trainable=True
)

input_ = Input(shape=(max_len_input,))
x = embedding_layer(input_)
bidirectional = Bidirectional(LSTM(lstm_dim))
x1 = bidirectional(x)
output = Dense(7, activation='softmax')(x1)


x = embedding_layer(input_)
bidirectional = Bidirectional(LSTM(lstm_dim))
x1 = bidirectional(x)
#dropt_out = Dropout(rate=0.5)
#x1 = dropt_out(x1)
output = Dense(7, activation='softmax')(x1)


model = Model(inputs=input_, outputs=output)

# compile
model.compile(
  # regular categorical_crossentropy requires one_hot_encoding for the targets, sparse_categorical_crossentropy is used to don't use the conversion
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#print(np.shape(X_train))
#print(np.shape(Y_train))
#exit()

# train
print('Training model...')
r = model.fit(X_train, Y_train, batch_size=128, epochs=32, validation_split=0.2)


# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()


#test
results = model.evaluate(X_test,Y_test)
print("test loss, test acc:", results)

