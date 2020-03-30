from pathlib import Path
import spacy
import requests
from spacy.util import minibatch, compounding
from flask import Flask, json


api = Flask(__name__)
nlp = spacy.load("model")

@api.route('/ner/sentence/<string:user_id>', methods=['GET'])
def get_companies(user_id):
	response = {}
	entities = {}
	doc = nlp(user_id)
	

	response['sentence'] = user_id
	response['entities'] = {}
	
	for ent in doc.ents:
		entities[ent.text] = ent.label_
	response['entities'] = entities

	return json.dumps(response)

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0')
    




