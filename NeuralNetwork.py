import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from metrics import Metrics

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.metrics = Metrics()
        self.best_metric = 0
        self.patience = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self):
        raise NotImplementedError


    def train_step(self, i, data):
        with torch.no_grad():
            batch_x_text,  batch_y = (elem.cuda() for elem in data)

        self.optimizer.zero_grad()
        Xt_logit = self.forward(batch_x_text)
        loss = self.loss_func(Xt_logit, batch_y)
        loss.backward()
        self.optimizer.step()

        print('Batch[{}] - loss: {:.6f}'.format(i, loss.item()))
        return loss


    def fit(self, X_train, y_train, X_val, y_val):

        if torch.cuda.is_available():
            self.cuda()

        batch_size = self.config['batch_size']
        X_train = torch.LongTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_train, y_train)
        dataiter = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['reg'])

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch+1,"/", self.config['epochs'])
            avg_loss = 0
            avg_acc = 0

            self.train()
            for i, data in enumerate(dataiter):
                loss = self.train_step(i, data)

                if (i+1) % 50 == 0:
                    self.evaluate(X_val, y_val)
                    self.train()

                avg_loss += loss.item()
            cnt = y_train.size(0) // batch_size + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))

            self.evaluate(X_val, y_val)
            if epoch > 10 and self.patience >= 2 and self.config['lr'] >= 1e-5:
                self.load_state_dict(torch.load(self.config['save_path']))
                self.adjust_learning_rate()
                print("Decay learning rate to: ", self.config['lr'])
                print("Reload the best model...")
                self.patience = 0


    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.config['lr'] = param_group['lr']
            print("Now lr: ", param_group['lr'])


    def evaluate(self, X_val,  y_val):
        y_pred, y_pred_top = self.predict(X_val)
        OE, HL, MacroF1, MicroF1 = self.metrics.calculate_all_metrics(y_val, y_pred, y_pred_top)
        metric = MacroF1 - HL

        if metric > self.best_metric:
            self.best_metric = metric
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            # pickle.dump(self.user_embedding.weight.data.cpu().numpy(),
            #             file=open("data/user_embedding"+str(epoch)+".pkl", 'wb'), protocol=4)
            print("save model!!!")
        else:
            self.patience += 1
        print("Val set metric:", metric)
        print("Best val set metric:", self.best_metric)


    def predict(self, X_test):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        X_test = torch.LongTensor(X_test).to(self.device)
        dataset = TensorDataset(X_test)
        dataiter = DataLoader(dataset, batch_size=200)

        y_pred = []
        y_pred_top = []
        for i, data in enumerate(dataiter):
            batch_x_text = data[0]
            logit = self.forward(batch_x_text)
            predicted = logit > 0.5

            _, predicted_top = torch.max(logit, dim=1)
            y_pred_top += predicted_top.data.cpu().numpy().tolist()
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred, y_pred_top

