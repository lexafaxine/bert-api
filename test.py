import json
import bert_ner
import pickle
import time
import requests
import os


def preprocess(text, label_map, max_seq_length, tokenizer):
    text = text.split(' ')
    # label = text.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(text):
        # print("word= ", word)
        token = tokenizer.tokenize(word)
        if '▁' in token:
            token.remove('▁')
        # print("token= ", token)
        tokens.extend(token)
        for m in range(len(token)):
            # dummy
            labels.append("O")

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("[PAD]")

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    # feature = InputFeatures(
    #     input_ids=input_ids,
    #     input_mask=input_mask,
    #     segment_ids=segment_ids,
    #     label_ids=label_ids,
    # )

    with open("token_pred.txt", "w", encoding='utf-8') as writer:
        for token in ntokens:
            if token != "[PAD]":
                writer.write(token + '\n')

    return {
        "instances": [
            {"input_ids": input_ids,
             "input_mask": input_mask,
             "segment_ids": segment_ids,
             "label_ids": label_ids
             }
        ]
    }


def visualize(input_ids, result_ids, tokenizer, label_map):
    assert len(input_ids) == len(result_ids) == 128
    word_list = tokenizer.convert_ids_to_tokens(input_ids)
    # convert result_ids to label_list:
    label_list = []

    for result in result_ids:
        if result == 0:
            break
        label = label_map[result]
        label_list.append(label)

    output = []
    output_cell = []
    n = len(label_list)

    # label to type

    d = {"O": "Nontype", "X": "Cross", "I-ORG": "Organization", "B-ORG": "Organization", "B-DAT": "Date", "I-DAT": "Date",
         "B-ART": "Artifact", "I-ART": "Artifact", "B-MNY": "Money", "I-MNY": "Money", "B-TIM": "Time",
         "I-TIM": "Time", "B-PNT": "Percent", "I-PNT": "Percent", "B-PSN": "Person", "I-PSN": "Person",
         "B-LOC": "Location", "I-LOC": "Location"}

    now = None

    cell_type = None
    cell_content = []

    for i in range(n):
        if word_list[i] == "[SEP]":
            break
        elif word_list[i] == '[CLS]':
            continue

        else:
            word = word_list[i]
            if word[0] == "#":
                # is a subword
                word = word[2:]

            label = label_list[i]
            part = d[label]

            if label[0] == "B":
                if cell_type is not None:
                    cell = {
                        "type": cell_type,
                        "content": cell_content
                    }
                    output_cell.append(cell)
                    cell_content = []

                cell_type = d[label]
                cell_content.append(word)

            elif label[0] == "I" or label == "X":
                # couldnt be a end of a ner
                cell_content.append(word)

            elif label == "O":
                if cell_type == d[label]:
                    cell_content.append(word)
                else:
                    if cell_type is not None:
                        cell = {
                            "type": cell_type,
                            "content": cell_content
                        }
                        output_cell.append(cell)
                        cell_content = []

                    cell_type = d[label]
                    cell_content.append(word)
            else:
                raise ValueError

    if cell_type is not None:
        cell = {
            "type": cell_type,
            "content": cell_content
        }
        output_cell.append(cell)

    for cell in output_cell:
        cell["content"] = "".join(cell["content"])

    return output_cell


def predict(label_map, feature):
    # predict
    start = time.time()
    resp = requests.post('http://localhost:8501/v1/models/ner_32k:predict',
                         json=feature)
    end = time.time()
    return resp.json()


if __name__ == "__main__":
    tokenizer = bert_ner.tokenization_ja.MecabBertTokenizer("vocab.txt",
                                                            mecab_dict_path="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    #   lable map: lable to id
    with open('./output/label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}

    text = "韓国の文在寅（ムンジェイン）大統領は18日、新年の記者会見を開いた。8日にソウル中央地裁が日本政府に対し、元慰安婦の女性に賠償するよう命じる判決を出したことについて、「正直、混乱している」と述べた。"

    input = preprocess(text, label2id, 128, tokenizer)

    print(input)

    predictions = predict(id2label, input)

    print(predictions)

    input_ids = input["instances"][0]["input_ids"]
    result_ids = predictions['predictions'][0]

    output = visualize(input_ids, result_ids, tokenizer, id2label)

    print("============================================================")

    print(output)
