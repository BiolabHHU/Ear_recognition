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
# from VGGVox import MainModel


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class AutoEncoderModel(pl.LightningModule):
    def __init__(self, input_size=13, hidden_size=256, encoded_size=512,
                 dropout_rates=[0.9, 0.2], batch_size=128, learning_rate=0.001, num_classes=60, data_folder_path='D:/yds/ear_recognition'):
        super().__init__()

        # Updated to match the new input feature size (13 * 299)
        self.encoder = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(input_size * 200, hidden_size),  # 13 * 299
            nn.ReLU(),
            nn.Linear(hidden_size, encoded_size),
            nn.ReLU()
        )

        # Updated to match the new input feature size (13 * 299)
        self.decoder1 = nn.Linear(encoded_size, 256)  # 13 * 299
        self.decoder2 = nn.Linear(256, num_classes)

        # self.output = nn.Linear(input_size * 299, batch_size)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mse_loss = nn.MSELoss()
        self.r2_metric = torchmetrics.R2Score()

        self.dataset = Dataset(data_folder_path=data_folder_path)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=60)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.save_hyperparameters()

    def forward(self, x):
        # x shape: (batch_size, 13, 299)
        x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, 13 * 299)
        encoded = self.encoder(x)
        encoded = self.decoder1(encoded)
        decoded = self.decoder2(encoded)
        return decoded

    def extract_x_vec(self, x):

        x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, 13 * 299)
        x = self.encoder(x)
        x_vec = self.decoder1(x)
        return x_vec


    # 训练模型
    # def training_step(self, batch, batch_index):
    #     samples, labels, _ = batch
    #     outputs = self(samples.float())
    #     # 打印 outputs 和 labels 的形状
    #     print(f"Outputs shape: {outputs.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #
    #     loss = self.mse_loss(outputs, labels.float())
    #     r2 = self.r2_metric(outputs, labels)
    #     self.log('train_loss', loss)
    #     self.log('train_r2', r2)
    #     return {'t_loss': loss}
    def training_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'train_preds': outputs, 'train_labels': labels, 'train_id': id}


    # 用记录器记录训练损失和准确性
    def training_step_end(self, outputs):
        self.log('train_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['train_preds'], outputs['train_labels'])
        self.log('train_step_acc', self.accuracy, prog_bar=True)
        return {'loss': outputs['loss'], 'acc': accuracy}

    # 为日志记录器创建图形和直方图
    def training_epoch_end(self, outputs):
        if self.current_epoch == 0:
            sample = torch.rand((1, 200, 13))
            self.logger.experiment.add_graph(AutoEncoderModel(), sample)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # 计算验证数据的损失，以检查是否过拟合
    # def validation_step(self, batch, batch_index):
    #     samples, labels, _ = batch
    #     outputs = self(samples.float())
    #     # 打印 outputs 和 labels 的形状
    #     print(f"Outputs shape: {outputs.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #
    #     loss = self.mse_loss(outputs, labels.float())
    #     r2 = self.r2_metric(outputs, labels)
    #     self.log('val_loss', loss)
    #     self.log('val_r2', r2)
    #     return {'v_loss': loss}
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
        # data = []
        # for i in range(len(outputs['val_labels'])):
        #     data.append(
        #         [outputs['val_labels'][i].item(), outputs['val_preds'][i].argmax().item(), outputs['val_id'][i]])
        #
        # with open('D:/PyCharmPython/Speaker-Recognition-x-vectors-main/validation_data_voice.csv', 'a',
        #           newline='') as file:
        #     writer = csv.writer(file)
        #     if self.current_epoch == 0:
        #         writer.writerow(['True Label', 'Predicted Label', 'ID'])
        #     writer.writerows(data)
        return {'loss': outputs['loss'], 'acc': accuracy}

    # 这里的测试步骤不用作测试步骤！
    # 相反，它被用来提取x向量
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

    # 只加载训练数据
    def train_dataloader(self):
        self.dataset.load_data(train=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    # Load only the validation data
    def val_dataloader(self):
        self.dataset.load_data(val=True)
        val_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return val_data_loader

    # 同时加载训练和验证或测试数据来提取x向量
    # 在“训练”模式下提取x向量用于PLDA训练，在“测试”模式下用于测试PLDA
    def test_dataloader(self):
        if extract_mode == 'train':
            self.dataset.load_data(train=True, val=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4,
                                          shuffle=False)
        if extract_mode == 'test':
            self.dataset.load_data(test=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4,
                                          shuffle=False)
        return test_data_loader


if __name__ == "__main__":

    # 调整模型、PLDA、训练等参数。在这里
    # 在这里设置您自己的数据文件夹路径！
    # VoxCeleb MUSAN和RIR必须在同一数据/目录中！
    # 也可以通过调整，只执行程序的选定部分：
    # train_x_vector_model, extract_x_vectors, train_plda and test_plda
    # 当只运行程序的后期部分时，必须给出一个checkpoint_path，并且程序的早期部分必须至少执行一次
    print('setting up model and trainer parameters')
    config = Config(data_folder_path='D:/yds/ear_recognition',
                    # checkpoint_path='D:/PyCharmPython/Speaker-Recognition-x-vectors-main/testlogs/lightning_logs/version_1_ecapa_tdnn_teacher/checkpoints/last.ckpt',
                    checkpoint_path='none',
                    train_x_vector_model=1,
                    extract_x_vectors=1,
                    train_plda=1,
                    test_plda=1,
                    x_vec_extract_layer=6,
                    plda_rank_f=200)  # TODO delete most of this

    # 定义模型和Trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="testlogs/")
    # early_stopping_callback = EarlyStopping(monitor="val_step_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_step_loss', save_top_k=10, save_last=True, verbose=True)

    if config.checkpoint_path == 'none':
        model = AutoEncoderModel(
            # learning_rate=config.learning_rate,
            # batch_size=config.batch_size,

            data_folder_path=config.data_folder_path)
    else:
        model = AutoEncoderModel.load_from_checkpoint(config.checkpoint_path)

    # 初始化数据集，给定训练和测试路径，并分配标签
    model.dataset.init_samples_and_labels()

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         logger=tb_logger,
                         log_every_n_steps=1,
                         # accelerator='cpu',#TODO delete
                         accelerator='cpu',
                         max_epochs=config.num_epochs)
    # small test adjust options: fast_dev_run=True, limit_train_batches=0.0001, limit_val_batches=0.001, limit_test_batches=0.002

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
    for i in range(1, 4):
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
                                                   threshold=-5)
            print(f'FAR: {far}, FRR: {frr}, Accuracy: {accuracy}')
            eer_list.append(eer)
    avg_eer = sum(eer_list) / len(eer_list)
    print('Average EER: ', avg_eer)
    Average_accuracy = 1 - 2 * avg_eer
    print('Average Accuracy: ', Average_accuracy)
