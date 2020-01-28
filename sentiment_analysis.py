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
import tensorflow_hub as hub

from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.backend import clear_session
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece



class SentimentModel:

	def __init__(self):

		#LOAD MODELS
		self.SentimentModel = pickle.load(open("model/train.pkl", "rb"))
		self.TFsession, self.embedded_text, self.text_input = self.initializeTfSession()


	def initializeTfSession(self):
		# Create graph and finalize.
		g = tf.Graph()
		with g.as_default():
			text_input = tf.placeholder(dtype=tf.string, shape=[None])
			embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
			embedded_text = embed(text_input)
			init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
			g.finalize()

		# session created and initialized.
		session = tf.Session(graph=g)
		session.run(init_op)
		return (session, embedded_text, text_input)

	def get_scores(self, text):

		text = self.TFsession.run(self.embedded_text, feed_dict={self.text_input: [text]})

		predictions = self.SentimentModel.predict(text)
		predictions = pd.DataFrame(predictions,columns=['negative','someWhat negative','neutral','someWhat positive','positive'])
		predictions = predictions.to_json(orient='records')[1:-1].replace('},{', '} {')
		return predictions
		

