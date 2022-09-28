# Simple emotion classification considering the isear dataset with 7 emotions (joy, fear, anger, sadness, disgust, shame, and guilt)
# using Bidirectional LSTM, dropout and a dense layer1


'''
For f1-score
'binary': Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
'samples': Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

#nltk.download('punkt')


max_num_words = 20000
embedding_dim = 300
lstm_dim = 150


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
#print("inputs[0]:", inputs[0])


# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
#for line in open('../emotion_embeddings/embeddings/senti-embedding/emb_nrc_vad_300dim_2.txt'):
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

# lstm_dim=168
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
output = Dense(7, activation='softmax', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x1)


x = embedding_layer(input_)
bidirectional = Bidirectional(LSTM(lstm_dim))
x1 = bidirectional(x)
output = Dense(7, activation='softmax')(x1)# softmax

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
r = model.fit(X_train, Y_train, batch_size=128, epochs=32, validation_split=0.2, verbose=0)#32

pred = model.predict(X_test, verbose=1)
print(pred[0])
pred = np.argmax(pred, axis=1)
print(pred[0])
exit()
y_test_ = [np.argmax(y, axis=0) for y in y_test]
pred = [np.argmax(y, axis=0) for y in pred]

precision = precision_score(y_true=y_test_, y_pred=pred, average='binary')
recall = recall_score(y_true=y_test_, y_pred=pred, average='binary')
acc = accuracy_score(y_true=y_test_, y_pred=pred)
r2 = r2_score(y_test_, pred)
f1 = f1_score(y_true=y_test_, y_pred=pred, average='macro')

print('acc: ', acc)
print('r2: ', r2)
#print('precision: ', precision)
#print('recall: ', recall)
print('f1: ', f1)
print('------------------------------------------')
