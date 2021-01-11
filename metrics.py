
import numpy as np
from sklearn.metrics import hamming_loss, f1_score


class Metrics():

    def __init__(self):
        super(Metrics, self).__init__()

    def one_error(self, y_true, y_pred_top):
        cnt = 0
        for i, y in enumerate(y_pred_top):
            if y_true[i, y] != 1:
                cnt += 1
        return cnt/len(y_true)


    def calculate_all_metrics(self, y_true, y_pred, y_pred_top):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        OE = self.one_error(y_true, y_pred_top)
        HL = hamming_loss(y_true, y_pred)
        MacroF1 = f1_score(y_true, y_pred, average='macro')
        MicroF1 = f1_score(y_true, y_pred, average='micro')

        print("OE: ", OE)
        print("HL: ", HL)
        print("MacroF1: ", MacroF1)
        print("MicroF1: ", MicroF1)
        return OE, HL, MacroF1, MicroF1

