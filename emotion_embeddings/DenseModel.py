from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


class DenseModel(Model):
	def __init__(self, num_units, activation_function='tanh'):
		super(DenseModel, self).__init__()
		# define all layers in init
		self.dense_layer = Dense(num_units, activation=activation_function, 
			kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))
		self.output_layer = Dense(3, activation='linear')

	def call(self, inputs, training=True):
		dense = self.dense_layer(inputs)

		return self.output_layer(dense)

