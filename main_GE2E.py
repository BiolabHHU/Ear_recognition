import os
import csv
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.metric_stats import calculate_FAR_FRR
import plda_classifier as pc
from config import Config
from dataset import Dataset

# from plda_score_stat import plda_score_stat_object
from tdnn_layer import TdnnLayer
from VGGVox import MainModel


class XVectorModel(pl.LightningModule):
    def __init__(self,
                 input_size=13,
                 hidden_size=512,
                 x_vec_extract_layer=6,
                 dropout_p=0.0,
                 x_vector_size=128,
                 learning_rate=0.001,
                 num_classes=60,
                 batch_size=512,
                 batch_norm=True,
                 nmels=13,
                 hidden=256,
                 num_layer=3,
                 proj=512,
                 data_folder_path='D:/yds/ear_recognition'):
        super().__init__()
        # self.speaker_encoder = MainModel
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = Dataset(data_folder_path=data_folder_path)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=60)
        self.save_hyperparameters()
        self.save_dir = 'D:/PyCharmPython/Speaker-Recognition-x-vectors-main'

        self.LSTM_stack = nn.LSTM(nmels, hidden, num_layers=num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hidden, proj)

        self.fc1 = nn.Linear(512, 128)  # 可选的全连接层
        self.fc2 = nn.Linear(128, 60)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        x = F.relu(self.fc1(x))  # (batch, 128)
        x = F.relu(self.fc2(x))  # (batch, num_classes)
        # x = F.softmax(x, dim=1)
        return x

    def extract_x_vec(self, x):  # 定义了提取X-vector的方法，根据x_vec_extract_layer参数指定提取的层，并返回相应的X-vector。

        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        x_vec = self.fc1.forward(x)
        return x_vec

    def save_attention_weights(self, attention_weights):
        # 转换为 numpy 数组并减少维度
        attention_weights_np = attention_weights.detach().cpu().numpy()

        # 平铺二维数据，例如按(batch_size, seq_len)
        # 假设 seq_length 是第二个维度
        attention_weights_np = attention_weights_np.reshape(attention_weights_np.shape[0], -1)

        # 将 numpy 数组转换为 DataFrame
        df = pd.DataFrame(attention_weights_np)

        # 保存为 CSV 文件
        file_path = os.path.join(self.save_dir, 'attention_weights_voice.csv')
        df.to_csv(file_path, mode='w', header=True, index=False)  # 追加模式保存
    # The statistic pooling layer

    # Train the model
    def training_step(self, batch, batch_index):  # 定义了训练步骤，包括获取样本数据、前向传播计算损失并返回结果。
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'train_preds': outputs, 'train_labels': labels, 'train_id': id}

    # Log training loss and accuracy with the logger
    def training_step_end(self, outputs):  # 定义了训练步骤结束后的处理，包括记录训练损失和准确率。
        self.log('train_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['train_preds'], outputs['train_labels'])
        self.log('train_step_acc', self.accuracy, prog_bar=True)
        return {'loss': outputs['loss'], 'acc': accuracy}

    # Create graph and histogram for the logger
    def training_epoch_end(self, outputs):
        if (self.current_epoch == 0):
            sample = torch.rand((1, 223, 13))
            self.logger.experiment.add_graph(XVectorModel(), sample)

        for name, params in self.named_parameters():

            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # Calculate loss of validation data to check if overfitting
    def validation_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'val_preds': outputs, 'val_labels': labels, 'val_id': id}

    # Log validation loss and accuracy with the logger
    def validation_step_end(self, outputs):
        self.log('val_step_loss', outputs['loss'], prog_bar=True)
        accuracy = self.accuracy(outputs['val_preds'], outputs['val_labels'])
        self.log('val_step_acc', self.accuracy, prog_bar=True)
        data = []
        # for i in range(len(outputs['val_labels'])):
        #     data.append(
        #         [outputs['val_labels'][i].item(), outputs['val_preds'][i].argmax().item(), outputs['val_id'][i]])
        #
        # with open('D:/PyCharmPython/Speaker-Recognition-x-vectors-main/validation_data_voice.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     if self.current_epoch == 0:
        #         writer.writerow(['True Label', 'Predicted Label', 'ID'])
        #     writer.writerows(data)
        return {'loss': outputs['loss'], 'acc': accuracy}

    def test_step(self, batch, batch_index):
        samples, labels, id = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels, id)]

    # 在生成了所有的x向量之后，将它们附加到预定义的列表中
    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label, id in batch_output:
                for x, l, i in zip(x_vec, label, id):
                    x_vector.append((i, int(l.cpu().numpy()), np.array(x.cpu().numpy(), dtype=np.float64)))
        return test_step_outputs

    # 配置优化器，使用了Adam优化器 (torch.optim.Adam)，并传递了模型的参数(self.parameters())以及学习率 (lr=self.learning_rate)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Load only the training data
    def train_dataloader(self):
        self.dataset.load_data(train=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    # Load only the validation data
    def val_dataloader(self):
        self.dataset.load_data(val=True)
        val_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return val_data_loader

    def test_dataloader(self):
        if (extract_mode == 'train'):
            self.dataset.load_data(train=True, val=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4,
                                          shuffle=False)
        if (extract_mode == 'test'):
            self.dataset.load_data(test=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4,
                                          shuffle=False)
        return test_data_loader


if __name__ == "__main__":

    print('setting up model and trainer parameters')
    config = Config(data_folder_path='D:/yds/ear_recognition',
                    checkpoint_path='none',
                    # checkpoint_path='D:/PyCharmPython/Speaker-Recognition-x-vectors-main/testlogs/lightning_logs/version_10/checkpoints/last.ckpt',
                    train_x_vector_model=0,
                    extract_x_vectors=0,
                    train_plda=1,
                    test_plda=1,
                    x_vec_extract_layer=6,
                    plda_rank_f=100)  # TODO delete most of this

    # Define model and trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="testlogs/")
    # early_stopping_callback = EarlyStopping(monitor="val_step_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_step_loss', save_top_k=10, save_last=True, verbose=True)

    if config.checkpoint_path == 'none':
        model = XVectorModel(input_size=config.input_size,
                             hidden_size=config.hidden_size,
                             num_classes=config.num_classes,
                             x_vector_size=config.x_vector_size,
                             x_vec_extract_layer=config.x_vec_extract_layer,
                             batch_size=config.batch_size,
                             learning_rate=config.learning_rate,
                             batch_norm=config.batch_norm,
                             dropout_p=config.dropout_p,
                             data_folder_path=config.data_folder_path)

    else:
        model = XVectorModel.load_from_checkpoint(config.checkpoint_path)
    model.dataset.init_samples_and_labels()

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         logger=tb_logger,
                         log_every_n_steps=1,
                         accelerator='cpu',  # TODO delete
                         # accelerator='gpu', devices=[0],
                         max_epochs=config.num_epochs)
    # small test adjust options: fast_dev_run=True, limit_train_batches=0.0001, limit_val_batches=0.001, limit_test_batches=0.002
    # Train the x-vector model

    # 训练x-vector模型
    if config.train_x_vector_model:
        print('training x-vector model')
        if config.checkpoint_path == 'none':
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=config.checkpoint_path)

    # 提取x-vectors
    if config.extract_x_vectors:
        print('extracting x-vectors')
        if not os.path.exists('x_vectors'):
            os.makedirs('x_vectors')
        # 提取用于训练PLDA分类器的x-vectors并保存到.csv
        x_vector = []
        extract_mode = 'train'
        if config.train_x_vector_model:
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_train_v1_5_l7relu.csv')  # TODO set to default name
        elif config.checkpoint_path != 'none':
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_train_v1_5_l7relu.csv')  # TODO set to default name
        else:
            print('could not extract train x-vectors')

        # 提取用于测试PLDA分类器的x-vectors并保存到.csv
        x_vector = []
        extract_mode = 'test'
        if (config.train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_test_v1_5_l7relu.csv')  # TODO set to default name
        elif (config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_test_v1_5_l7relu.csv')  # TODO set to default name
        else:
            print('could not extract test x-vectors')

    eer_list = []
    for i in range(1, 11):
        x_vectors_train = None
        x_vectors_test = None
        if config.train_plda:
            print('loading x_vector data')
            if not os.path.exists('plda'):
                os.makedirs('plda')
            # Extract the x-vectors, labels and id from the csv
            # x_vectors_data = pd.read_csv('x_vectors/x_vector_test_v1_5_l7relu.csv').iloc[840:1680]
            x_vectors_data = pd.read_csv('x_vectors/x_vector_test_v1_5_l7relu.csv').iloc[0:900]
            # x_vectors_data = pd.read_csv('x_vectors/x_vector_test_v1_5_l7relu.csv').iloc[0:27]
            # 将数据集按照每个人的数据进行分组
            grouped_data = x_vectors_data.groupby(x_vectors_data.index // 30)
            train_data = []
            test_data = []
            for name, group in grouped_data:
                # 将每个人的数据分成训练集和测试集
                train_group, test_group = train_test_split(group, test_size=0.1, random_state=i)
                train_data.append(train_group)
                test_data.append(test_group)
            # 将训练集和测试集合并为一个数据集
            x_vectors_train = pd.concat(train_data)
            x_vectors_test = pd.concat(test_data)
            x_id_train = np.array(x_vectors_train.iloc[:, 1])
            x_label_train = np.array(x_vectors_train.iloc[:, 2], dtype=int)
            x_vec_train = np.array(
                [np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_train.iloc[:, 3]])
            print('generating x_vec stat objects')
            tr_stat = pc.get_train_x_vec(x_vec_train, x_label_train, x_id_train)

            # Train plda
            print('training plda')
            plda = pc.setup_plda(rank_f=50, nb_iter=10)
            plda = pc.train_plda(plda, tr_stat)
            pc.save_plda(plda, 'plda_ivec_v2_d50')

            # Train plda
            print('training plda')
            plda = pc.setup_plda(rank_f=100, nb_iter=10)
            plda = pc.train_plda(plda, tr_stat)
            pc.save_plda(plda, 'plda_ivec_v2_d100')
            '''
            # Train plda
            print('training plda')
            plda = pc.setup_plda(rank_f=150, nb_iter=10)
            plda = pc.train_plda(plda, tr_stat)
            pc.save_plda(plda, 'plda_ivec_v2_d150')
            # Train plda
            print('training plda')
            plda = pc.setup_plda(rank_f=200, nb_iter=10)
            plda = pc.train_plda(plda, tr_stat)
            pc.save_plda(plda, 'plda_ivec_v2_d200')
            '''
        if (config.test_plda):
            # Extract the x-vectors, labels and id from the csv
            print('testing plda')
            if (not config.train_plda):
                plda = pc.load_plda('plda/plda_ivec_v2_d100.pickle')  # TODO set to default name
            # x_vectors_test = pd.read_csv('x_vectors/x_vector_train_v1_5_l7relu.csv', skiprows=range(1, 757), nrows=84)
            x_id_test = np.array(x_vectors_test.iloc[:, 1])
            te_label = np.array(x_vectors_test.iloc[:, 2], dtype=int)
            x_vec_test = np.array(
                [np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_test.iloc[:, 3]])
            te_stat = pc.get_x_vec_stat(x_vec_test, x_id_test)

            # x_vectors_train = pd.read_csv('x_vectors/x_vector_train_v1_5_l7relu.csv').head(756)
            x_id_train = np.array(x_vectors_train.iloc[:, 1])
            en_label = np.array(x_vectors_train.iloc[:, 2], dtype=int)
            x_vec_train = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in
                                    x_vectors_train.iloc[:, 3]])  # TODO set to default name
            en_stat = pc.get_x_vec_stat(x_vec_train, x_id_train)

            scores_plda = pc.plda_scores(plda, en_stat, te_stat)

            # Dividing scores into positive and negative
            # Dividing scores into positive and negative
            positive_scores = []
            negative_scores = []
            for en in en_label:
                for te in te_label:
                    if (en == te):
                        positive_scores.append(1)
                        negative_scores.append(0)
                    else:
                        positive_scores.append(0)
                        negative_scores.append(1)
            positive_scores_mask = np.array(positive_scores, dtype=bool)
            negative_scores_mask = np.array(negative_scores, dtype=bool)
            positive_scores_mask = np.reshape(positive_scores_mask, (len(en_label), len(te_label)))
            negative_scores_mask = np.reshape(negative_scores_mask, (len(en_label), len(te_label)))
            positive_scores = scores_plda.scoremat[positive_scores_mask]
            negative_scores = scores_plda.scoremat[negative_scores_mask]

            # Calculating EER
            print('calculating EER')
            eer, eer_th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
            print('EER: ', eer)
            print('threshold: ', eer_th)
            far, frr, accuracy = calculate_FAR_FRR(torch.tensor(positive_scores), torch.tensor(negative_scores),
                                                   threshold=0)
            print(f'FAR: {far}, FRR: {frr}, Accuracy: {accuracy}')
            eer_list.append(eer)
    avg_eer = sum(eer_list) / len(eer_list)
    print('Average EER: ', avg_eer)
    Average_accuracy = 1 - 2 * avg_eer
    print('Average Accuracy: ', Average_accuracy)