# libs import
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras
import logging as log
import json

# Intializing TF session and graph
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'to_dict'):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)

class SentimentModel:

	def __init__(self):


		self.SentimentModel = keras.models.load_model('model/train.h5') #loading model
		self.TFsession, self.embedded_text, self.text_input = self.initializeTfSession()  #initiating tf graph


	def initializeTfSession(self):
		# Create graph and finalize.
		# try:
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

	def model_predict(self,emb_text):
		# using created tf session and graph as default
		try:
			with session.as_default():
				with session.graph.as_default():
					
					predictions = self.SentimentModel.predict(emb_text)
					return predictions
		except Exception as ex:
			log.log('Internal TF bakend Internal Error', ex, ex.__traceback__.tb_lineno)
   
	def get_scores(self,text):

		text =[text.lower()]
		emb_text = self.TFsession.run(self.embedded_text, feed_dict={self.text_input: text})
		predictions = self.model_predict(emb_text)
		predictions = pd.DataFrame(predictions,columns=['negative','someWhat negative','neutral','someWhat positive','positive'])
		predictions = predictions.to_json(orient='records')[1:-1].replace('},{','} {')
		
		return predictions