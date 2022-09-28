import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score

from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import word_tokenize

embedding_dim = 300


df = pd.read_csv('../util/isear.csv',header=None)
# Remove 'No response' row value in isear.csv
df = df[~df[1].str.contains("NO RESPONSE")]
#print(df[0].unique())
#exit()

input_sequences = []
for val in df[1]:
	input_sequences.append(word_tokenize(val))


dict_vocab = {}
word2idx = {}
counter = 1
for sent in input_sequences:
	for word in sent:
		if word not in word2idx:
			word2idx[word] = counter
			counter += 1
		if word in dict_vocab:
			dict_vocab[word] += 1
		else:
			dict_vocab[word] = 1

input2idx = []
for sent in input_sequences:
	vec = []
	for word in sent:
		vec.append(word2idx[word])
	input2idx.append(vec)

df['class_int'] = pd.Categorical(df[0]).codes
#pd.Categorical(df[0]).categories[int_label]
print('Found %s unique input tokens.' % len(dict_vocab))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)
print('max_len_input', max_len_input)
exit()
inputs = pad_sequences(input2idx, maxlen=max_len_input)
# when padding is not specified it takes the default at the begining of the sentence
#print("inputs.shape:", inputs.shape)

# store all the pre-trained word vectors
word2vec = {}
lexico = 'nrc_vad'
#for line in open('../emotion_embeddings/embeddings/senti-embedding-modif/emb_' + lexico + '_%ddim_2.txt' % embedding_dim):
#for line in open(os.path.join('../util/glove.6B.%sd.txt' % embedding_dim)):
for line in open('../util/ewe_uni.txt'):
	values = line.split()
	word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print("Number of word embeddings: ", len(word2vec))

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(word2idx) + 1
embedding_matrix = np.zeros((len(inputs), max_len_input, embedding_dim))
#print(num_words)
for word, i in word2idx.items():
	#print(i)
	embedding_vector = word2vec.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all zeros.
		embedding_matrix[i] = embedding_vector
	else:
		embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)



# Perform one-hot encoding on df[0] i.e emotion
#enc = OneHotEncoder()#handle_unknown='ignore')
#outputs = enc.fit_transform(np.array(df[0]).reshape(-1,1)).toarray()
outputs = list(df['class_int'])


# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

print('logistic regression:')
lr = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=500)
lr.fit(x_train, y_train)

pred = lr.predict(x_test)
#print(pred)
#exit()
#pred = np.where(pred > 0.5, 1, 0)

#y_test_ = [np.argmax(y, axis=0) for y in y_test]
#pred = [np.argmax(y, axis=0) for y in pred]

precision = precision_score(y_true=y_test, y_pred=pred, average='macro')
recall = recall_score(y_true=y_test, y_pred=pred, average='macro')
f1 = f1_score(y_true=y_test, y_pred=pred, average='macro')
acc = accuracy_score(y_true=y_test, y_pred=pred)
r2 = r2_score(y_true=y_test, y_pred=pred)


#print('Lexico: ', lexico)
#print('Emo_emb_size: ', lstm_dim_vec)
print('acc: ', acc)
print('precision: ', precision)
print('recall: ', recall)
print('f1: ', f1)
#print('r2: ', r2)
print('------------------------------------------')

print('SVM:')
svm = SVC(random_state=0)
svm.fit(x_train, y_train)

pred = svm.predict(x_test)
#pred = np.where(pred > 0.5, 1, 0)

#y_test_ = [np.argmax(y, axis=0) for y in y_test]
#pred = [np.argmax(y, axis=0) for y in pred]

precision = precision_score(y_true=y_test, y_pred=pred, average='macro')
recall = recall_score(y_true=y_test, y_pred=pred, average='macro')
f1 = f1_score(y_true=y_test, y_pred=pred, average='macro')
acc = accuracy_score(y_true=y_test, y_pred=pred)
r2 = r2_score(y_true=y_test, y_pred=pred)


#print('Lexico: ', lexico)
#print('Emo_emb_size: ', lstm_dim_vec)
print('acc: ', acc)
print('precision: ', precision)
print('recall: ', recall)
print('f1: ', f1)
#print('r2: ', r2)