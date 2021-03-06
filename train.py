from numpy import datetime64, datetime_data
import BERT_model
import hyperparameters
import tokenizer
import os
import tensorflow as tf
from tensorboard_output import Metrics
from tensorflow import keras
from datetime import datetime

train_x = []
train_y = []
val_x = []
val_y = []
classes = [
    "作曲",
    "歌手",
    "出品公司",
    "导演",
    "妻子",
    "丈夫",
    "出版社",
    "作者",
    "主演",
    "出生日期",
    "国籍",
    "民族",
    "目",
    "成立日期",
    "出生地",
    "连载网站",
    "所属专辑",
    "身高",
    "编剧",
    "母亲",
    "毕业院校",
    "作词",
    "字",
    "创始人",
    "号",
    "朝代",
    "父亲",
    "嘉宾",
    "上映时间",
    "所在城市",
    "改编自",
    "海拔",
    "简称",
    "注册资本",
    "占地面积",
    "祖籍",
    "总部地点",
    "制片人",
    "面积",
    "人口数量",
    "首都",
    "主持人",
    "董事长",
    "主角",
    "气候",
    "修业年限",
    "专业代码",
    "官方语言",
    "邮政编码"
]


def create_f1():
    def f1_function(y_true, y_pred):
        y_pred_binary = tf.where(y_pred >= 0.5, 1., 0.)
        y_true = tf.cast(y_true, dtype=tf.float32)
        tp = tf.reduce_sum(y_true * y_pred_binary)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        possible_positives = tf.reduce_sum(y_true)
        return tp, predicted_positives, possible_positives
    return f1_function


class F1_score(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # handles base args (e.g., dtype)
        self.f1_function = create_f1()
        self.tp_count = self.add_weight("tp_count", initializer="zeros")
        self.all_predicted_positives = self.add_weight(
            'all_predicted_positives', initializer='zeros')
        self.all_possible_positives = self.add_weight(
            'all_possible_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, predicted_positives, possible_positives = self.f1_function(
            y_true, y_pred)
        self.tp_count.assign_add(tp)
        self.all_predicted_positives.assign_add(predicted_positives)
        self.all_possible_positives.assign_add(possible_positives)

    def result(self):
        precision = self.tp_count / self.all_predicted_positives
        recall = self.tp_count / self.all_possible_positives
        f1 = 2*(precision*recall)/(precision+recall)
        return f1


# read dataset
target_rows = 1000000
f = open("train.txt", encoding="utf-8")
line = f.readline()
i = 1
while line and i < target_rows:
    line = f.readline()
    if not line:
        break
    sentence_front_part = line.split("\t")[0]
    sentence_classes = sentence_front_part.split(",")
    sentence = line.split("\t")[1]
    train_x.append(sentence)
    y_array = []
    for single_class in classes:
        if single_class in sentence_classes:
            y_array.append(1)
        else:
            y_array.append(0)
    train_y.append(y_array)
    i += 1
f.close()
print("Preprocessing train data...")
train_x, train_y = tokenizer.get_model_data(train_x, train_y)
print("Preprocess completely...")
print(train_x[1:10])
print(train_y[1:10])
print("tarin x shape:", len(train_x[0]))
f = open("test.txt", encoding="utf-8")
line = f.readline()
i = 1
while line and i < target_rows:
    line = f.readline()
    if not line:
        break
    sentence_front_part = line.split("\t")[0]
    sentence_classes = sentence_front_part.split(",")
    sentence = line.split("\t")[1]
    val_x.append(sentence)
    y_array = []
    for single_class in classes:
        if single_class in sentence_classes:
            y_array.append(1)
        else:
            y_array.append(0)
    val_y.append(y_array)
    i += 1
f.close()
print("Preprocessing valit data...")
val_x, val_y = tokenizer.get_model_data(val_x, val_y)
print("Preprocess completely...")
print(val_x[1:10])
print(val_y[1:10])
# train model
logdir = os.path.join("log", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
args = hyperparameters.args
model = BERT_model.create_model(
    args['bert_model_name'], args['label_num'])
model.compile(loss='binary_crossentropy',
              optimizer="adam", metrics=[F1_score()])
model.fit(train_x, train_y, epochs=args["epoch"], verbose=1,
          callbacks=[Metrics(valid_data=(val_x, val_y)), tensorboard_callback],
          batch_size=args["batch_size"],
          validation_data=(val_x, val_y),
          validation_batch_size=args["batch_size"])

model_path = os.path.join("output", "model", "mulclassifition.h5")
model.save_weights(model_path)

pbmodel_path = os.path.join("ouput", "tfmodel", "output")
tf.keras.models.save_model(
    model, pbmodel_path, save_format="tf", overwrite=True)
