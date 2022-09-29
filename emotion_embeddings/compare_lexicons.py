import pandas as pd
import settings


df_nrc_vad = pd.read_csv(settings.input_dir_lexicon_vad + 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', keep_default_na=False, header=None, sep='\t')
df_nrc_ei = pd.read_csv(settings.input_dir_lexicon + 'NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt', keep_default_na=False, header=None, sep='\t')
df_nrc_el = pd.read_csv(settings.input_dir_lexicon + 'NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', keep_default_na=False, header=None, sep='\t')

dict_nrc_vad =  list(df_nrc_vad[0])
dict_nrc_ei =  list(df_nrc_ei[0])
dict_nrc_el =  list(df_nrc_el[0])

dict_nrc_vad = [val.lower() for val in dict_nrc_vad]
dict_nrc_ei = [val.lower() for val in dict_nrc_ei]
dict_aux = []
for val in dict_nrc_el:
	if val.lower() not in dict_aux:
		dict_aux.append(val.lower())
dict_nrc_el = dict_aux

print('Len nrc_vad: ', len(dict_nrc_vad))
#anger, anticipation, disgust, fear, joy, sadness, surprise, and trust
print('Len nrc_ei: ', len(dict_nrc_ei))
#(anger, fear, anticipation, trust, surprise, sadness, joy, or disgust) or one of two polarities (negative or positive)
print('Len nrc_el: ', len(dict_nrc_el))

count_not_in_nrc_ei = 0
count_not_in_nrc_el= 0

for val in dict_nrc_vad:
	if val not in dict_nrc_ei:
		count_not_in_nrc_ei += 1
	if val not in dict_nrc_el:
		count_not_in_nrc_el += 1

print('-----------------------------------------------------------------------------')
print('For NRC_VAD')
print(count_not_in_nrc_ei, ' value not in nrc_ei')
print(count_not_in_nrc_el, ' value not in nrc_el')


count_not_in_nrc_vad = 0

for val in dict_nrc_ei:
	if val not in dict_nrc_vad:
		count_not_in_nrc_vad += 1

print('-----------------------------------------------------------------------------')
print('For NRC_EI')
print(count_not_in_nrc_vad, ' value not in nrc_vad')