import pandas as pd
import numpy as np
import pymssql
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import tensorflow_hub as hub

from keras.utils import to_categorical
from sklearn.metrics import accuracy_score


data = pd.read_csv('train.tsv', sep='\t')

test_data = pd.read_csv('test.tsv',sep='\t')

"""
Movie Review dataset for sentiment analysis
"""
"""
The sentiment labels are:

0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

"""

print(tf.version.VERSION)

# universal encoder word embeddings

# for multilingual
# "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"


g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

# session created and initialized.
session = tf.Session(graph=g)
session.run(init_op)

def word_embeddings(input_text):
    emb = session.run(embedded_text, feed_dict={text_input: input_text})
    return emb


emb_phrases = word_embeddings(data['Phrase'])


# feature matrix
x = emb_phrases
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
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))# input
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))# input
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))# input
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))# input
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal',input_dim=512))# input
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
                                # validation_data=(x_val, y_val))



def get_scores(UserId,post):

    post = word_embeddings([post])
    y_pred = model.predict(post)
    predictions = pd.DataFrame(y_pred,columns=['negative','someWhat negative','neutral','someWhat positive','positive'])
    
    return predictions

# get_scores(1,post="how are you")
# text = "Are there any activity groups around Coventry"

# get_scores(text)


def get_intent_label(pred):

    result = np.amax(pred)
    index = np.where(pred == np.amax(pred))
    index = list(index[1])
    # intent = idx2intent[index[0]]

    return index




"""data cleaning function must be added in the pipeline before infering the results (mostly not required)"""
# text = "amazing"
# emb_text = word_embeddings([text])
# pred = model.predict(emb_text)

# y_pred = get_intent_label(y_pred) 

# evaluate model
train_acc = model.evaluate(xtrain,y_train_binary)
test_acc = model.evaluate(xtest, y_test_binary)


# sentiment analysis predictions for Gro en-GB locale
def set_conn():
    conn= pymssql.connect(host='35.242.167.76',
                                    database='GroCommunityDB_en-GB',
                                    user='groDBeditUser',
                                    password='cwy6KpE8QBM5Cjzc',
                                )
    
    return conn

conn = set_conn()
posts = pd.read_sql_query('''SELECT* FROM dbo.posts''', conn)

post_id = posts[['Id']]
post_id.columns = ['PostId']
user_id = posts[['UserId']]

emb_posts = word_embeddings(posts['Body'])

posts_sentiment = model.predict(emb_posts)



posts_sentiment = pd.DataFrame(posts_sentiment,columns=['negative','someWhat negative','neutral','someWhat positive','positive'])

posts_sentiment = pd.concat([post_id, user_id, posts_sentiment],axis=1)


posts_sentiment.to_csv('post_sentiment_en-GB')

 


