# libs
from sentiment_analysis import get_scores
from io import StringIO    
import json
import flask
from flask import Flask, request
import time
from flask import jsonify

def __init__(self,UserId,post):
    self.UserId = UserId
    self.post = post


def get_final_scores(json):
    scores = get_scores(json['UserId'],json['post'])
    return scores


app = Flask(__name__)

@app.route('/ping',methods=['GET'])
@app.route('/',methods=['POST'])


def sentiment_scores():
    if flask.request.content_type == 'application/json':
        input_json = flask.request.get_json()
        print("Input json")
        print(input_json)
    else:
        return flask.Response(response='Content type should be application/json', status=415, mimetype='application/json')
    response = get_final_scores(input_json)

# Get the response
    out = StringIO()    
    json.dump(response, out)
    return flask.Response(response=out.getvalue(), status=200, mimetype='application/json')

    
if __name__ == '__main__':
       
    app.run(port=8000)




