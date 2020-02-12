import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from tensorflow.keras.layers import Layer
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import preprocessing
import tensorflow_hub as hub
from keras.utils import to_categorical
import keras
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from bert_embedding import BertEmbedding
import pandas as pd

# LOAD DATA
data = pd.read_csv('train.tsv',sep='\t')

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

phrases = data['Phrase'].astype(str)
# bert-uncased-model for multi-lingual word embeddings
data = merge_classes(data)
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased',batch_size=10000)


sentences_slice = phrases[:100]
emb_sent = bert_embedding(sentences_slice)	

features = np.array(emb_sent)

# bert_embedding dump
pickle.dump(emb_sentences,open("WordEmbeddings/bert_embeddings.pkl","wb"))

# feature matrix
x = features
y = data['Sentiment'][:100]

# train-test split
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,random_state=True)



# preprocessed labels
y_train_binary = to_categorical(ytrain) #converting the labels to binary to avoid shape errors regarding the target variable
y_test_binary = to_categorical(ytest)


"""
Convolutional Neural Network
"""
def conv1D():
	# Convolution Neural Network
	# # Initialising the CN
	classifier = Sequential()

	# Convolution1D layer_1
	classifier.add(Convolution1D(filters=128, kernel_size=3,input_shape = (769,1),kernel_initializer='uniform' ,activation = 'relu'))


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
	classifier.add(Dense(2, activation='softmax', kernel_initializer='random_normal')) # target shape

	# Compiling the CNN
	classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

	classifier.summary()
	return classifier

# padding sequence 
xtrain = np.expand_dims(xtrain, axis=2) 

xtest = np.expand_dims(xtest,axis=2)

conv1D = conv1D()
history = conv1D.fit(xtrain, y_train_binary,batch_size=100,epochs=5, validation_data=(xtest,y_test_binary),verbose=1)

# loading model
cnn_model = keras.models.load_model('model/cnn_model.h5')

"""
CONFUSION MATRIX & F-1 SCORE
"""
y_pred = cnn_model.predict(xtest)

matrix = confusion_matrix(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1))

from sklearn.metrics import f1_score
# f1_score
score = f1_score(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
print("f-1_score: {}".format(score))

# CM_matrix
cm_matrix = pd.DataFrame(matrix,
                    columns=['negative','somewhat negative','neutral','someWhat positive','positive'],
                    index=['negative','somewhat negative','neutral','someWhat positive','positive'])


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

conv1D.save('model/cnn_model2.h5')


# Correctly Classified 
# 440+2711+16865+2716+695 = 23427


# Misclassified 
# 171648


print("\(-_-)/")



