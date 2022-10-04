import pandas as pd
import re
import numpy as np
from string import punctuation
from sklearn.preprocessing import MinMaxScaler


punctuation = re.sub(r'[-.]', '', punctuation)


def preprocessing(dict_input):
	dict_modif = {}
	for key, value in dict_input.items():
		#del d[key]
		if re.search(r'\d', key) == None and re.search(r'[' + punctuation + ']', key) == None:
			dict_modif[key] = value
		#if re.search(r'[' + punctuation + ']', key):
		#	print(key)


	return dict_modif

def normalization(dict_input):
	new_dict = {}
	list_values = list(dict_input.keys())
	matrix = []
	for key in list_values:
		matrix.append(dict_input[key])

	matrix = np.asarray(matrix)
	minmax_scale = MinMaxScaler(feature_range=(0, 1))
	matrix = minmax_scale.fit_transform(matrix)

	idx = 0
	for key in list_values:
		new_dict[key] = matrix[idx]
		idx += 1

	return new_dict

def manyArgs(*arg):
  print "I was called with", len(arg), "arguments:", arg


def create_vocabulary(list_vocab_1, list_vocab_2, list_vocab_3, list_vocab_4):
	list_vocab_1 = set(list_vocab_1)
	list_vocab_2 = set(list_vocab_2)
	list_vocab_3 = set(list_vocab_3)
	list_vocab_4 = set(list_vocab_4)

	return list_vocab_1.union(list_vocab_2, list_vocab_3, list_vocab_4)


print('Loading DepecheMood ...')
df = pd.read_csv('/home/carolina/corpora/lexicons/depeche_mood_2014/DepecheMood_normfreq.txt', keep_default_na=False, sep='\t')

dict_dm = {}
emotions_dm = [val.lower() for val in list(df.columns)[1:9]]
print('emotions: ', emotions_dm)
for index, row in df.iterrows():
	dict_dm[re.sub(r'#.*', '', str(row[0]))] = [float(row[i]) for i in range(1,9)]
print('original_size: ', len(dict_dm))
dict_dm = preprocessing(dict_dm)
print('mosified_size: ', len(dict_dm))
list_vocab_1 =  list(dict_dm.keys())



print('---------------------------------------------------')
# missing test removing '#' from punctuation
print('NRC hashtag emotion lexicon ...')
# <AffectCategory><tab><term><tab><score>
emotions_hel = []
dict_hel = {}
df = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Hashtag-Emotion-Lexicon/NRC-Hashtag-Emotion-Lexicon-v0.2.txt', keep_default_na=False, sep='\t', header=None)
for index, row in df.iterrows():
	if str(row[0]) not in emotions_hel:
		emotions_hel.append(str(row[0]))

	if str(row[1]) in dict_hel:
		arr_val = dict_hel[str(row[1])]
	else:
		arr_val = np.zeros((8))
	arr_val[emotions_hel.index(str(row[0]))] = float(row[2])
	dict_hel[str(row[1])] = arr_val

print('emotions: ', emotions_hel)
print('original size: ', len(dict_hel))
dict_hel = preprocessing(dict_hel)
print('mosified_size: ', len(dict_hel))
dict_hel = normalization(dict_hel)
list_vocab_2 =  list(dict_hel.keys())


print('---------------------------------------------------')
print('Loading NRC Emotion Intensity Lexicon ...')
df = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt', keep_default_na=False, header=None, sep='\t')
emotions_int = ['anger', 'fear', 'sadness', 'joy']
dict_data_emo_int = {}
for index, row in df.iterrows():
	#if str(row[1]) not in emotions_int:
	#	emotions_int.append(str(row[1]))

	if str(row[1]) in emotions_int:
		if str(row[0]) in dict_data_emo_int:
			arr_val = dict_data_emo_int[str(row[0])]
		else:
			arr_val = np.zeros((8))
		arr_val[emotions_int.index(str(row[1]))] = float(row[2])
		dict_data_emo_int[str(row[0])] = arr_val

print('emotions: ', emotions_int)
print('size: ', len(dict_data_emo_int))
list_vocab_3 =  list(dict_data_emo_int.keys())




print('---------------------------------------------------')
print('Loading NRC Emotion Lexicon ...')
df = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', keep_default_na=False, header=None, sep='\t')
emotions_emo_lex = []
dict_data_emo_lex = {}
idx_emo = 0
for index, row in df.iterrows():
	if str(row[1]) not in emotions_emo_lex:
		emotions_emo_lex.append(str(row[1]))

	if str(row[0]) in dict_data_emo_lex:
		arr_val = dict_data_emo_lex[str(row[0])]
	else:
		arr_val = np.zeros((10))
	arr_val[emotions_emo_lex.index(str(row[1]))] = int(row[2])
	dict_data_emo_lex[str(row[0])] = arr_val

print('emotions: ', emotions_emo_lex)
print('size: ', len(dict_data_emo_lex))
list_vocab_4 =  list(dict_data_emo_lex.keys())


print('---------------------------------------------------')
vocabulary = create_vocabulary(list_vocab_1, list_vocab_2, list_vocab_3, list_vocab_4)
print(len(vocabulary))

