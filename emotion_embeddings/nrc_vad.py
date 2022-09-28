import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk import TweetTokenizer

from gensim.models import KeyedVectors

import matplotlib.pyplot as plt



df = pd.read_csv('/home/carolina/corpora/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', sep='\t', header=None)











