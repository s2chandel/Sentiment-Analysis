# libs
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


def __init__(self, text):

    self.text = text


# Model Inference
def sentiment_scores(json):
    """
    calls get_scores function in the
    SentimentModel class
    """
    scores = sm.get_scores(json['text'])
    return scores


app = Flask(__name__)


@app.route('/ping', methods=['GET'])
@app.route('/', methods=['POST'])
def sentiment_analysis():

    if flask.request.content_type == 'application/json':
        input_json = flask.request.get_json()
        print("Input json")
        print(input_json)
    else:
        return flask.Response(response='Content type should be application/json', status=415, mimetype='application/json')

    # Get the response
    response = sentiment_scores(input_json)

    out = StringIO()
    json.dump(response, out)
    return flask.Response(response=out.getvalue(), status=200, mimetype='application/json')


if __name__ == '__main__':

    app.run(port=8000)
