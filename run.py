import pickle
import torch
from metrics import Metrics
from sklearn.metrics import f1_score
import numpy as np
from KRF import KRF
from baselines.TextCNN import TextCNN
from baselines.TextLSTM import TextLSTM
from baselines.MLP import MLP
from baselines.HAN_LCM import HAN_LCM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def loadData(task):
    path = "./data/" + task + "/"
    X_train_label, X_train_comment,  word_embeddings = pickle.load(open(path + "X_train.pkl", mode='rb'))
    X_valid_label, X_valid_comment = pickle.load(open(path + "X_valid.pkl", mode='rb'))
    X_test_label, X_test_comment = pickle.load(open(path + "X_test.pkl", mode='rb'))

    config["C"] = 22 if task == 'douban_music' else 20
    config["task"] = task
    config['adj_file'] = path + "adj_file.pkl"
    config['max_comments'] = len(X_train_comment[0])
    config['max_len'] = len(X_train_comment[0][0])
    config['embedding_weights'] = word_embeddings

    return  X_train_label, X_train_comment,  \
            X_valid_label, X_valid_comment,  \
            X_test_label, X_test_comment


def train_and_test(model_class, task):
    model_suffix = model_class.__name__.lower().strip("text")
    config['save_path'] = 'checkpoints/weights.best.'+ model_suffix

    X_train_label, X_train_comment, \
    X_valid_label, X_valid_comment, \
    X_test_label, X_test_comment = loadData(task)

    model = model_class(config)
    # model.fit(X_train_comment, X_train_label,
    #           X_valid_comment, X_valid_label)

    print("================================================")
    model.load_state_dict(state_dict=torch.load(config['save_path']))
    y_pred, y_pred_top = model.predict(X_test_comment)

    metric = Metrics()
    metric.calculate_all_metrics(X_test_label, y_pred, y_pred_top)

    if task == 'douban_music':
        X_test_label = np.array(X_test_label)
        y_pred = np.array(y_pred)
        F1_top1 = f1_score(X_test_label, y_pred, labels=[1], average = 'micro')
        F1_top2 = f1_score(X_test_label, y_pred, labels=[0], average = 'micro')
        F1_top3 = f1_score(X_test_label, y_pred, labels=[2], average = 'micro')
        F1_top4 = f1_score(X_test_label, y_pred, labels=[3], average = 'micro')
        F1_top5 = f1_score(X_test_label, y_pred, labels=[4], average = 'micro')

        F1_few1 = f1_score(X_test_label, y_pred, labels=[13], average='micro')
        F1_few2 = f1_score(X_test_label, y_pred, labels=[17], average='micro')
        F1_few3 = f1_score(X_test_label, y_pred, labels=[18], average='micro')
        F1_few4 = f1_score(X_test_label, y_pred, labels=[20], average='micro')
        F1_few5 = f1_score(X_test_label, y_pred, labels=[21], average='micro')

        print('--'*20)
        print("F1_top1: ", F1_top1)
        print("F1_top2: ", F1_top2)
        print("F1_top3: ", F1_top3)
        print("F1_top4: ", F1_top4)
        print("F1_top5: ", F1_top5)
        print('--' * 20)
        print("F1_few1: ", F1_few1)
        print("F1_few2: ", F1_few2)
        print("F1_few3: ", F1_few3)
        print("F1_few4: ", F1_few4)
        print("F1_few5: ", F1_few5)


config = {
    'lr':1e-3,
    'reg':0,
    'alpha':1.0,
    'batch_size':128,
    'maxlen':1000,
    'nb_filters': 100,
    'dropout': 0.5,
    'input_dim':300,
    'epochs':20,
}


if __name__ == '__main__':
    model = KRF # KRF  TextCNN   TextLSTM  MLP  HAN_LCM
    train_and_test(model, task="douban_music")
    # train_and_test(model, task="amazon_music")

