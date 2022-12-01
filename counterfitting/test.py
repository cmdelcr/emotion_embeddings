import os
import re


pos_emo_counter = []
neg_emo_counter = []
with open('/home/carolina/git/emotion_embeddings/counterfitting/constraints/posEmotionList.csv', 'r') as file:
	for line in file:
		pos_emo_counter.append(re.sub(r' ', '_', re.sub(r'\n', '', line)))

with open('/home/carolina/git/emotion_embeddings/counterfitting/constraints/negEmotionList.csv', 'r') as file:
	for line in file:
		neg_emo_counter.append(re.sub(r' ', '_', re.sub(r'\n', '', line)))


pos_emo_gen = []
neg_emo_gen = []
with open('/home/carolina/git/emotion_embeddings/retrofitting/true_emotions.txt', 'r') as file:
	for line in file:
		pos_emo_gen.append(re.sub(r'\t', '_', re.sub(r'\n', '', line)))

with open('/home/carolina/git/emotion_embeddings/retrofitting/opposite_emotions.txt', 'r') as file:
	for line in file:
		neg_emo_gen.append(re.sub(r'\t', '_', re.sub(r'\n', '', line)))

print('Positive ')
print('orig: ', len(pos_emo_counter), ' gen:', len(pos_emo_gen))
print('Negative ')
print('orig: ', len(neg_emo_counter), ' gen:', len(neg_emo_gen))
if len(pos_emo_counter) == len(pos_emo_gen):
	print('equal lenght positive emotions')
if len(neg_emo_counter) == len(neg_emo_gen):
	print('equal lenght negative emotions')