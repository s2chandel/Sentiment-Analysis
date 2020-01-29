import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

import pickle
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece


data = pd.read_csv('train.tsv', sep='\t')

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
pkl.dump(emb,open("WordEmbeddings/embbeddings.pkl","wb"))

# loading embeddings
emb_x = pickle.load(open("WordEmbeddings/embbeddings.pkl", "rb"))

# feature matrix
x = emb_x
y = data['Sentiment']

# train-test split
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,random_state=True)



# preprocessed labels
y_train_binary = to_categorical(ytrain) #converting the labels to binary to avoid shape errors regarding the target variable
y_test_binary = to_categorical(ytest)

#sequential neural network
def model():
    model = Sequential()
    # adding dropout layers to regularise the neural network
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))# input
    model.add(Dense(5, activation='softmax', kernel_initializer='random_normal')) # target shape
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',)
    model.summary()
    return model

model = model()
model.fit(xtrain, y_train_binary,
            epochs=20,
            batch_size=100,verbose=1)


train_acc =  model.evaluate(xtrain,y_train_binary)
print("train_accuray:{}".format(train_acc))


test_acc = model.evaluate(xtest,y_test_binary)
print("test_accuray:{}".format(test_acc))


model.save('model/train.h5')


# pkl.dump(model,open("model/train.pkl","wb"))






























