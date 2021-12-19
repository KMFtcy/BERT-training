from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertTokenizer, BertModel, BertForMaskedLM


dataset = ["天将降大任于私人也", "放下包袱开动机器"]

bert_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
for data in dataset:
    output = bert_tokenizer.encode(data)
    print(data)
    print(output)
# model = BertModel.from_pretrained()


num_labels = 10
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels)
