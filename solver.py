
from utils.utils import *
from model.FADFD import FADFD
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings
import pandas as pd
# import torch.nn as nn
warnings.filterwarnings('ignore')




        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='test', dataset=self.dataset)

        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def build_model(self):
        self.model = FADFD(p=self.p,select=self.select)
        
        if torch.cuda.is_available():
            self.model.cuda()


    def test(self):
        # find the threshold
        attens_energy = []
        for i, (input_data, data_global, labels) in enumerate(self.thre_loader):

            input = input_data.float().to(self.device)  # (128,100,51)
            data_global = data_global.float().to(self.device)
            score = self.model(input, data_global)

            metric = score.unsqueeze(-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0)
        test_energy = np.array(attens_energy)

        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)


        test_labels = []
        attens_energy = []
        for i, (input_data, data_global, labels) in enumerate(self.thre_loader):

            input = input_data.float().to(self.device)  # (128,100,51)
            data_global = data_global.float().to(self.device)

            score = self.model(input, data_global)

            metric = score.unsqueeze(-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)


        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]

        input_show=np.concatenate([np.expand_dims(test_labels,axis=-1),attens_energy],axis=1)
        df = pd.DataFrame(input_show)
        excel_writer = pd.ExcelWriter('tensor_data.xlsx', engine='openpyxl')  # 选择 'xlsxwriter' 或 'openpyxl' 作为引擎
        df.to_excel(excel_writer, index=False, sheet_name='Sheet1')
        excel_writer.save()

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))



        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
