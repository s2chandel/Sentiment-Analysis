import pickle

import keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import (LSTM, Convolution1D, Dense, Dropout, Embedding,
                          Flatten, MaxPooling1D)
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer

# import tf_sentencepiece
data = pd.read_csv('train.tsv', sep='\t')

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

data =  merge_classes(data)


"""
neg= 0
somewhat_neg= 1
neutral= 2
someWhat_pos= 3
pos= 4
remanufactored classes = ['neg'=1,'neutral'=2,'pos'=3]
"""
# print(tf.version.VERSION)

# Graph set up.
g = tf.Graph()
with g.as_default():
	text_input = tf.placeholder(dtype=tf.string, shape=[None])
	embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
	embedded_text = embed(text_input)
	init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.Session(graph=g)
session.run(init_op)



def word_embeddings(phrases):
	for phrase in phrases:
		emb = session.run(embedded_text, feed_dict={text_input: phrase})
	return emb

phrases = data['Phrase']

emb = word_embeddings([phrases])

# word_embeddings dump
pickle.dump(emb,open("WordEmbeddings/embbeddings.pkl","wb"))

# loading embeddings
emb_x = pickle.load(open("WordEmbeddings/embbeddings.pkl", "rb"))

# feature matrix
x = emb_x
y = data['Sentiment']

# train-test split
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,random_state=True)

# x_train = np.expand_dims(xtrain,axis=2)
# x_test = np.expand_dims(xtest, axis=2)


# preprocessed labels
y_train_binary = to_categorical(ytrain,num_classes=3) #converting the labels to binary to avoid shape errors regarding the target variable
y_test_binary = to_categorical(ytest)

#sequential neural network
def model():
	model = Sequential()
	# adding dropout layers to regularise the neural network
	model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))
	model.add(Dropout(0.2))																	  # 512 input nodes	connected to dense layer_1 128 nodes
																# weights = 512*128 +128 = 65664 params

	model.add(Dense(128, activation='relu', kernel_initializer='random_normal',))# input
	model.add(Dropout(0.2))

	model.add(Dense(128, activation='relu', kernel_initializer='random_normal',))# input
	model.add(Dropout(0.2))

	model.add(Dense(128, activation='relu', kernel_initializer='random_normal',))# input
	model.add(Dropout(0.2))

	model.add(Dense(3, activation='softmax', kernel_initializer='random_normal')) # target shape 3dim
	model.compile(optimizer='adam',
				loss='categorical_crossentropy',metrics=['acc'])
	model.summary()
	return model

model = model()
history = model.fit(xtrain, y_train_binary,
				epochs=5,
				batch_size=50,validation_data=(xtest,y_test_binary),verbose=1)
y_pred = model.predict(xtest)


# f1_score==50%
score = f1_score(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
print("f-1_score: {}".format(score))

train_acc =  model.evaluate(xtrain,y_train_binary)
print("train_accuray:{}".format(train_acc))


val_acc = model.evaluate(xtest,y_test_binary)
print("test_accuray:{}".format(val_acc))


# model serialization
model.save('model/train.h5')


"""Model Vizualization"""
# Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


"""
Convolutional Neural Network
"""
def model1():
	# Convolution Neural Network


	# Initialising the CNN
	classifier = Sequential()

	classifier.add(Embedding(117045,500,input_length=512))
	# Convolution1D layer_1
	# classifier.add(Convolution1D(filters=128, kernel_size=3,input_shape = (512,1),kernel_initializer='uniform' ,activation = 'relu'))

	# MaxPooling
	classifier.add(MaxPooling1D(pool_size = 2))

	# Convolution1D layer_2
	classifier.add(Convolution1D(filters=128,kernel_size=3, activation = 'relu')) # 128 3*3 matrix  
	classifier.add(MaxPooling1D(pool_size =2)) # filter matrix shape for maxpooling 2*2


	# Convolution1D layer_3
	classifier.add(Convolution1D(filters=128,kernel_size=3, activation = 'relu')) # 128 3*3 matrix  
	classifier.add(MaxPooling1D(pool_size =2)) # filter matrix shape for maxpooling 2*2

	# Convolution1D layer_4
	classifier.add(Convolution1D(filters=128,kernel_size=3, activation = 'relu')) # 128 3*3 matrix  
	classifier.add(MaxPooling1D(pool_size =2)) # filter matrix shape for maxpooling 2*2

	# Dropout Layer
	classifier.add(Dropout(0.2))

	# Flattening
	classifier.add(Flatten())# Flattening for the output of the convolutional layer

	# Full connection

	# classifier.add(Dense(nodes = 100, activation = 'relu'))
	classifier.add(Dense(100, activation = 'relu'))
	classifier.add(Dense(3, activation='softmax', kernel_initializer='random_normal')) # target shape

	# Compiling the CNN
	classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

	classifier.summary()
	return classifier

# padding sequence 
xtrain = np.expand_dims(xtrain, axis=2) 

xtest = np.expand_dims(xtest,axis=2)

model1 = model1()
history = model1.fit(xtrain, y_train_binary,batch_size=100,epochs=5, validation_data=(xtest,y_test_binary),verbose=1)



model1 = model1.save('model/model1.h5')

# loading model
model1 = keras.models.load_model('model/model1.h5')

"""
CONFUSION MATRIX & F-1 SCORE
"""
y_pred = model1.predict(xtest)


matrix = confusion_matrix(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1))

# f1_score (percent of correctly classified instances in the test set)
score = f1_score(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
print("f-1_score: {}".format(score))


# CM_matrix
cm_matrix = pd.DataFrame(matrix,
                    columns=['negative','somewhat negative','neutral','someWhat positive','positive'],
                    index=['negative','somewhat negative','neutral','someWhat positive','positive'])
cm_matrix = pd.DataFrame(matrix,
                    columns=['negative','neutral','positive'],
                    index=['negative','neutral','positive'])


figure = plt.figure(figsize=(8, 8))
sns.heatmap(cm_matrix, annot=True,fmt='g',cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


"""
=======================================================================================================================
"""
# Performance Vizualition
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


history.history



# Correctly Classified 
# 440+2711+16865+2716+695 = 23427

ytest_df = pd.DataFrame(ytest)
len(ytest_df[ytest_df['Sentiment']=='0'])
len(ytest_df[ytest_df['Sentiment']=='1'])
len(ytest_df[ytest_df['Sentiment']=='2'])
# Misclassified 
# 171648



print("\(-_-)/")
