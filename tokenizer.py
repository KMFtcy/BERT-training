
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import tensorflow as tf
import hyperparameters


def get_model_data(data, labels, max_seq_len=128):
    tokenizer = BertTokenizer.from_pretrained(
        hyperparameters.args['bert_model_name'])
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "label": []
    }
    for i in range(len(data)):
        sentence = data[i]
        input_ids = tokenizer.encode(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Truncate all sentences.
        )
        sentence_length = len(input_ids)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences([input_ids],
                                                                  maxlen=max_seq_len,
                                                                  dtype="long",
                                                                  value=0,
                                                                  padding="post")
        input_ids = input_ids.tolist()[0]
        attention_mask = [1] * sentence_length + \
            [0] * (max_seq_len - sentence_length)
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["label"].append(labels[i])
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["label"]
    return x, y


x, y = get_model_data(["这是一个测试", "这是第二个测试"], ["好", "坏"])
print("x:", x)
print("y:", y)
