# Integrated Stacking
import tensorflow as tf
import keras
import tf_sentencepiece
import tensorflow_hub as hub
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax
 
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'model/model' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models



# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	output = Dense(3, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
n_members = 2
members = load_all_models(n_members)
model = define_stacked_model(members)

import pandas as pd
import numpy as np
from keras.utils import to_categorical
import pickle

data = pd.read_csv('train.tsv',sep='\t')
emb_x = pickle.load(open("WordEmbeddings/embbeddings.pkl", "rb"))

def merge_classes(data):
	"""
	merging classes begining from zero to
	avoid error during to_categorical 
	transformation as method expects 
	the classes to start from 0

	"""
	data['Sentiment'] =data['Sentiment'].astype(str).replace(r'1',r'0')
	data['Sentiment'] =data['Sentiment'].astype(str).replace(r'2',r'1')
	data['Sentiment'] =data['Sentiment'].astype(str).replace(r'3',r'2')
	data['Sentiment'] =data['Sentiment'].astype(str).replace(r'4',r'2')

	return data

df_merged = merge_classes(data)

inputX = np.asanyarray(emb_x)
inputy = df_merged['Sentiment']

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	# X = [inputX for _ in range(len(model.input))]
	# encode output data
	X = inputX
	inputy_enc = to_categorical(inputy)

	history = model.fit(X, inputy_enc, epochs=2, verbose=1)

	return history
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(inputX,inputy,random_state=True)


stacked_model = fit_stacked_model(stacked_model,xtrain,ytrain)

model.save('model/stacked_model.h5')

stacked_model = keras.models.load_model('model/stacked_model.h5')

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X)

y_pred = predict_stacked_model(stacked_model,xtest)
y_pred_df = pd.DataFrame(y_pred)









print("\(-_-)/")

