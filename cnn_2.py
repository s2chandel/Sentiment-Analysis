import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import preprocessing
from keras.utils import to_categorical
import keras
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

# LOAD DATA
df = pd.read_csv('train.tsv',sep='\t')

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

texts = df['Phrase'].astype(str)
# classes merged df
df_merged = merge_classes(df)

# cut off reviews after 500 words
max_len = 512 

# consider only the top 10000 words
max_words = 10000 

# import tokenizer with the consideration for only the top 500 words
tokenizer = preprocessing.text.Tokenizer(num_words=max_words) 
# fit the tokenizer on the texts
tokenizer.fit_on_texts(texts) 
# convert the texts to sequences
sequences = tokenizer.texts_to_sequences(texts) 

word_index = tokenizer.word_index
print('Found %s unique tokens. ' % len(word_index))

# sequence padding
data = pad_sequences(sequences, maxlen=max_len)
print('Data Shape: {}'.format(data.shape))

labels = np.asanyarray(df_merged['Sentiment'])
# data shuffle before train_test_split
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# feature matrix
x = data
y = labels

# train-test split
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,random_state=True)

# preprocessed labels
y_train_binary = to_categorical(ytrain) #converting the labels to binary to avoid shape errors regarding the target variable

y_test_binary = to_categorical(ytest)


# Model
def conv1D():
	# Convolution Neural Network
	# # Initialising the CN
	classifier = Sequential()
	# embedding layer
	classifier.add(Embedding(117045,500,input_length=512)) ## input_lenth = 512 similar to the model1 input

	# Convolution1D layer_1
	classifier.add(Convolution1D(filters=128,kernel_size=3, activation = 'relu')) # 128 3*3 matrix  
	classifier.add(MaxPooling1D(pool_size =2)) # filter matrix shape for maxpooling 2*2


	# Convolution1D layer_2
	classifier.add(Convolution1D(filters=128,kernel_size=3, activation = 'relu')) # 128 3*3 matrix  
	classifier.add(MaxPooling1D(pool_size =2)) # filter matrix shape for maxpooling 2*2

	# Convolution1D layer_3
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

from keras.callbacks import TensorBoard
# tensorboard --logdir=model_logs
tensorboard = TensorBoard(log_dir='./model_logs', histogram_freq=0,
                          write_graph=True, write_images=False)

conv1D = conv1D()
history = conv1D.fit(xtrain, y_train_binary,batch_size=100,epochs=5, validation_data=(xtest,y_test_binary),callbacks=[tensorboard],verbose=1)
"""
serialising model
"""
# saving model
conv1D.save('model/model2.h5')
# loading model
cnn_model2 = keras.models.load_model('model/cnn_model2.h5')

"""
CONFUSION MATRIX & F-1 SCORE
"""
y_pred = cnn_model2.predict(xtest)

matrix = confusion_matrix(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1))

from sklearn.metrics import f1_score
# f1_score
score = f1_score(y_test_binary.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
print("f-1_score: {}".format(score))

# CM_matrix
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

conv1D.save('model/cnn_model2.h5')




print("\(-_-)/")



