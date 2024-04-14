from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
app = Flask(__name__)

model = './sm-bert-model'
tokenizer = './tokenizer'

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

@app.route("/")
def hello():
  return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)
    sentiment = int(torch.argmax(result.logits))+1

    response = jsonify({'sentiment': sentiment})
    
    return response

if __name__ == "__main__":
  app.run()