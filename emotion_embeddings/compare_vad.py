import pandas as pd

df_nrc = pd.read_csv('/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', keep_default_na=False, header=None, sep='\t')
df_anew = pd.read_csv('/home/carolina/corpora/lexicons/anew.csv', keep_default_na=False)
df_e_anew = pd.read_csv('/home/carolina/corpora/lexicons/e-anew.csv', keep_default_na=False)


dict_nrc =  list(df_nrc[0])
dict_anew =  list(df_anew['term'])
dict_e_anew =  list(df_e_anew['Word'])

dict_nrc = [val.lower() for val in dict_nrc]
dict_anew = [val.lower() for val in dict_anew]
dict_e_anew = [val.lower() for val in dict_e_anew]

print('Len nrc: ', len(dict_nrc))
print('Len anew: ', len(dict_anew))
print('Len e_anew: ', len(dict_e_anew))

count_not_in_anew = 0
count_not_in_e_anew = 0

for val in dict_nrc:
	if val not in dict_anew:
		count_not_in_anew += 1
	if val not in dict_e_anew:
		count_not_in_e_anew += 1

print('-----------------------------------------------------------------------------')
print('For NRC')
print(count_not_in_e_anew, ' value not in e_anew. ', count_not_in_anew, ' not in anew')


count_not_in_anew = 0
count_not_in_nrc = 0

for val in dict_e_anew:
	if val not in dict_anew:
		count_not_in_anew += 1
	if val not in dict_nrc:
		print(val)
		count_not_in_nrc += 1


print('-----------------------------------------------------------------------------')
print('\nFor E-ANEW')
print(count_not_in_nrc, ' value not in nrc. ', count_not_in_anew, ' not in anew')


count_not_in_e_anew = 0
count_not_in_nrc = 0

for val in dict_anew:
	if val not in dict_e_anew:
		count_not_in_e_anew += 1
	if val not in dict_nrc:
		count_not_in_nrc += 1


print('-----------------------------------------------------------------------------')
print('\nFor ANEW')
print(count_not_in_nrc, ' value not in nrc. ', count_not_in_e_anew, ' not in e-anew')