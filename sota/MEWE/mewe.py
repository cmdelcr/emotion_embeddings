import os
import re
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import create_emo_matrix_330

# dir for colab
#dir_name = '/content/drive/MyDrive/sota_embeddings/'
# dir for desktop
dir_name = '/home/carolina/embeddings/sota/mewe_embeddings/'


if not (os.path.isfile(dir_name + 'emo_embeddings_330.npy') and os.path.isfile(dir_name + 'keys.txt')):
	create_emo_matrix_330.create_matrix()

keys = []
embeddings = np.load(dir_name + 'emo_embeddings_330.npy')
for line in open(dir_name + 'keys.txt'):
	keys.append(re.sub(r'\n', '', line))

print('Size of embeddings: ', np.shape(embeddings))

pca = PCA(n_components=300)
embeddings = pca.fit_transform(embeddings)

with open(dir_name + 'emo_embeddings.txt', 'w') as f:
	i = 0
	for w_vec in embeddings:
		w_vec = w_vec.reshape(1, -1)
		f.write(keys[i].replace(" ", "_" ) + " ")
		np.savetxt(f, fmt='%.6f', X=w_vec)
		i += 1
	f.close()