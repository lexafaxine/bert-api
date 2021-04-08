import os
import gzip
import json
import pickle
import requests
import logging

from http import HTTPStatus
from flask import Flask, request, g
from visualize import visualize
from preprocess import preprocess


TF_SERVER_HOST = os.environ.get("TF_SERVER_HOST", "localhost")
TF_SERVER_PORT = int(os.environ.get("TF_SERVER_PORT", "8501"))
MECAB_DICT_PATH = os.environ["MECAB_DICT_PATH"]


def load_id2label():
    if 'label2id' not in g:
        with open(os.path.join("data", "label2id.pkl.gz"), 'rb') as rf:
            drf = gzip.decompress(rf.read())
            label2id = pickle.loads(drf)
            id2label = {value: key for key, value in label2id.items()}
            
            g.label2id = label2id
            g.id2label = id2label
    
    return g.label2id, g.id2label


def load_tokenizer():
    if 'tokenizer' not in g:
        from bert_japanese.tokenization import BertJapaneseTokenizerForPretraining
        fn = os.path.join("data", "vocab.txt")
        tokenizer = BertJapaneseTokenizerForPretraining(fn, mecab_dict_path=MECAB_DICT_PATH)
        g.tokenizer = tokenizer
    
    return g.tokenizer


def fetchPredictionResult(instances):
    logger = logging.getLogger()
    payload = {
        "instances": instances
    }
    
    for i in range(3):
        res = requests.post(f'http://{TF_SERVER_HOST}:{TF_SERVER_PORT:d}/v1/models/ner_32k:predict', json=payload)

        if res.status_code != HTTPStatus.OK:
            logger.warn(
                f"Try to fetch prediction result but failed with [{res.status_code}] in retry {i}")
            continue

        data = res.json()
        return data['predictions']
    else:
        return None
    
    
def create_app():
    app = Flask(__name__, static_url_path='')
    return app

app = create_app()


@app.route('/api/predict', methods=['POST'])
def predict():
    logger = logging.getLogger()
    
    if request.method == "POST":
        payload = json.loads(request.data)
        content = payload["content"]
        
        tokenizer = load_tokenizer()
        label2id, id2label = load_id2label()

        # preprocess
        instances = preprocess(content, label2id, max_seq_length=128, tokenizer=tokenizer)
        logger.info(instances)
        n = len(instances)
        # predict
        predictions = fetchPredictionResult(instances)
        
        if predictions is None:
            return "", HTTPStatus.INTERNAL_SERVER_ERROR
        
        assert n == len(predictions)
        
        # output
        results = [visualize(r["input_ids"], p, tokenizer, id2label) for p, r in zip(predictions, instances)]
        return {
            "result": results
        }
