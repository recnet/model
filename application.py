from flask import Flask, Response, request
from flask_cors import CORS
import json

class EndpointAction(object):
    """ Helper class for endpoints """
    def __init__(self, action):
        self.action = action
        self.response = Response(status=200,
                                 headers={'Content-Type' : 'application/json'})

    def __call__(self, *args):
        self.response.data = self.action()
        return self.response

class Application(object):
    """ A very simple JSON API to serve predictions """
    def __init__(self, model, session):
        self.model = model
        self.session = session
        self.app = Flask('application')
        CORS(self.app)
        self.app.add_url_rule('/predict',
                              'predict',
                              EndpointAction(self.make_prediction))

        print('Starting application...')
        self.app.run()


    def make_prediction(self):
        """ Make a prediction """
        text_to_predict = request.args.get('text')
        subreddit = request.args.get('subreddit')
        if subreddit is None:
            subreddit = 'UNK'
        predictions = self.model.predict(text_to_predict, subreddit)
        result = {'predicted':text_to_predict,
                  'subreddit':subreddit,
                  'predictions':predictions}
        return json.dumps(result)
