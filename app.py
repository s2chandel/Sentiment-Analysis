# libs
# from sentiment_analysis import get_scores
from io import StringIO    
import json
import flask
from flask import Flask, request
import time
from flask import jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

from sentiment_analysis import SentimentModel

sentiment_analysis_objects = {}
sm = SentimentModel()
# SentimentModel().get_scores("hi")

# class sentiment_analysis:
# def __init__(self,text):
# 	self.text = text

def reply(text):
	response =sm.get_scores(text)

	return response


def get_score(json):

    response = reply(json['text'])

    return response



app = Flask(__name__)

@app.route('/ping',methods=['GET'])
@app.route('/',methods=['POST'])


def sentiment_analysis():

   if flask.request.content_type == 'application/json':
       input_json = flask.request.get_json()
       print("Input json")
       print(input_json)
   else:
       return flask.Response(response='Content type should be application/json', status=415, mimetype='application/json')

   # Get the response
   response = get_score(input_json)

   out = StringIO()
   json.dump(response, out)
   return flask.Response(response=out.getvalue(), status=200, mimetype='application/json')



if __name__ == '__main__':
       
    app.run(port=8000)




