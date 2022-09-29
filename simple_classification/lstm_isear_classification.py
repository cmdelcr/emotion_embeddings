import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn import preprocessing
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from gensim.models import KeyedVectors

from nltk import TweetTokenizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


lstm_dim = 168
embedding_dim = 300
binary = True
epochs = 20
batch_size = 124#25
lstm_dim_arr = [3, 10, 30, 50, 100, 200, 300]
#lexicons = ['e-anew', 'nrc_vad']
lexicons = ['nrc_vad']


df = pd.read_csv('/home/carolina/corpora/emotion_datasets/isear.csv',header=None)
# Remove 'No response' row value in isear.csv
df = df[~df[1].str.contains("NO RESPONSE")]
df[2] = pd.Categorical(df[0]).codes

# convert numeric value back to string
#pd.Categorical(df[0]).categories[1]

train, test = train_test_split(df, test_size=0.2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train[1])
input_sequences = tokenizer.texts_to_sequences(train[1])

# get the word to index mapping for input language
word2idx = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

x_train = pad_sequences(input_sequences, maxlen=max_len_input)
# when padding is not specified it takes the default at the begining of the sentence


x_test = pad_sequences(tokenizer.texts_to_sequences(test[1]), maxlen=max_len_input)


# Perform one-hot encoding on df[0] i.e emotion
enc = OneHotEncoder()#handle_unknown='ignore')
y_train = enc.fit_transform(np.array(train[2]).reshape(-1,1)).toarray()
y_test = enc.fit_transform(np.array(test[2]).reshape(-1,1)).toarray()

#print(y_train)

# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
lexico = 'nrc_vad'
lstm_dim_vec = 300
for line in open('../emotion_embeddings/embeddings/senti-embedding/emb_' + lexico + '_%ddim_scaled_extended.txt' % lstm_dim_vec):
#for line in open('../emotion_embeddings/embeddings/senti-embedding/emb_' + lexico + '_%ddim_2.txt' % lstm_dim_vec):
#for line in open(os.path.join('/home/carolina/corpora/embeddings/glove/glove.6B.%sd.txt' % embedding_dim)):
#for line in open('/home/carolina/corpora/embeddings/emotions_embedings/ewe_uni.txt'):
#for line in open('/home/carolina/corpora/embeddings/emotions_embedings/sawe-tanh-pca-100-glove.txt'):
	values = line.split()
	word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print("Number of word embeddings: ", len(word2vec))

# Load vectors directly from the file
#word2vec = KeyedVectors.load_word2vec_format('/home/carolina/corpora/embeddings/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(word2idx) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word2idx.items():
	embedding_vector = word2vec.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all zeros.
		embedding_matrix[i] = embedding_vector
	else:
		embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)


'''count_known = 0
count_unk = 0
for word, i in word2idx.items():
	try:
		embedding_vector = word2vec[word]
		embedding_matrix[i] = embedding_vector
		count_known += 1
	except:
		embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
		count_unk += 1

print('Word2vec loaded words: ', count_known)
print('Unknown words: ', count_unk)'''

embedding_layer = Embedding(
  num_words,
  embedding_dim,
  weights=[embedding_matrix],
  input_length=max_len_input,
  trainable=False
)


input_ = Input(shape=(max_len_input,))
x = embedding_layer(input_)
bidirectional = Bidirectional(LSTM(lstm_dim))
x1 = bidirectional(x)
output = Dense(7, activation='softmax', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x1)


model = Model(inputs=input_, outputs=output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

pred = model.predict(x_test, verbose=1)

y_test_ = [np.argmax(y, axis=0) for y in y_test]
pred = [np.argmax(y, axis=0) for y in pred]

precision = precision_score(y_true=y_test_, y_pred=pred, average='macro')
recall = recall_score(y_true=y_test_, y_pred=pred, average='macro')
f1 = f1_score(y_true=y_test_, y_pred=pred, average='macro')
acc = accuracy_score(y_true=y_test_, y_pred=pred)
r2 = r2_score(y_true=y_test_, y_pred=pred)


#print('Lexico: ', lexico)
#print('Emo_emb_size: ', lstm_dim_vec)
print('acc: ', acc)
print('precision: ', precision)
print('recall: ', recall)
print('f1: ', f1)
print('r2: ', r2)
print('------------------------------------------')

#with open('results.csv', 'a') as file:
#	file.write(',' + lexico + ',' + str(lstm_dim_vec) + ',' + str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')
#	file.close()
