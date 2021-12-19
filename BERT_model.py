from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import TFBertModel
import tensorflow as tf
import hyperparameters

args = hyperparameters.args

class BertMultiClassifier(object):
    def __init__(self, bert_model_name, label_num):
        self.label_num = label_num
        self.bert_model_name = bert_model_name

    def get_model(self):
        model = TFBertModel.from_pretrained(self.bert_model_name)
        print("model type:", type(model))
        input_ids = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name="attention_mask")

        outputs = model(input_ids, attention_mask=attention_mask)[1]
        cla_outputs = tf.keras.layers.Dense(
            self.label_num, activation='sigmoid')(outputs)
        model = tf.keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=[cla_outputs])
        return model


def create_model(bert_model_name, label_nums):
    model = BertMultiClassifier(bert_model_name, label_nums).get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['learning_rate'])
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=['accuracy', tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])   # metrics=['accuracy']
    return model
