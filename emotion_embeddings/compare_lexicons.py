import pandas as pd


df_nrc_vad = pd.read_csv('/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', keep_default_na=False, header=None, sep='\t')
df_nrc_ei = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt', keep_default_na=False, header=None, sep='\t')

dict_nrc_vad =  list(df_nrc_vad[0])
dict_nrc_ei =  list(df_nrc_ei[0])

dict_nrc_vad = [val.lower() for val in dict_nrc_vad]
dict_nrc_ei = [val.lower() for val in dict_nrc_ei]

print('Len nrc_vad: ', len(dict_nrc_vad))
print('Len nrc_ei: ', len(dict_nrc_ei))

count_not_in_nrc_ei = 0

for val in dict_nrc_vad:
	if val not in dict_nrc_ei:
		count_not_in_nrc_ei += 1

print('-----------------------------------------------------------------------------')
print('For NRC_VAD')
print(count_not_in_nrc_ei, ' value not in nrc_ei')


count_not_in_nrc_vad = 0

for val in dict_nrc_ei:
	if val not in dict_nrc_vad:
		count_not_in_nrc_vad += 1

print('-----------------------------------------------------------------------------')
print('For NRC_EI')
print(count_not_in_nrc_vad, ' value not in nrc_vad')