import copy
import logging
import time

import torch
from torch import nn

from fedml_api.distributed.fed_transformer.utils import transform_tensor_to_list, WarmupCosineSchedule, \
    WarmupLinearSchedule


class FedAVGTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict,
                 test_data_local_dict, train_data_num, device, model, args):
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=args.learning_rate,
                                             momentum=0.9,
                                             weight_decay=args.wd)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        if args.decay_type == "cosine":
            self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=args.warmup_steps,
                                                  t_total=args.comm_round)
        else:
            self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=args.warmup_steps,
                                                  t_total=args.comm_round)

        self.train_data_extracted_features, self.test_data_extracted_features= self._extract_features()

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.head.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def _extract_features(self):
        train_data_extracted_features = dict()
        test_data_extracted_features = dict()
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.train_local):
                time_start_test_per_batch = time.time()
                x = x.to(self.device)
                extracted_feature_x, _ = self.model.transformer(x)
                train_data_extracted_features[batch_idx] = (extracted_feature_x[:, 0].cpu().detach(), target)
                time_end_test_per_batch = time.time()
                logging.info("train_local feature extraction - time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_local):
                time_start_test_per_batch = time.time()
                x = x.to(self.device)
                extracted_feature_x, _ = self.model.transformer(x)
                test_data_extracted_features[batch_idx] = (extracted_feature_x[:, 0].cpu().detach(), target)
                time_end_test_per_batch = time.time()
                logging.info("test_local feature extraction - time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))

        return train_data_extracted_features, test_data_extracted_features

    def train(self):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx in self.train_data_extracted_features.keys():
                (x, labels) = self.train_data_extracted_features[batch_idx]
                x = x.to(self.device)
                labels = labels.to(self.device)
                # logging.info(images.shape)
                self.optimizer.zero_grad()
                log_probs = self.model.head(x)
                # print(log_probs.shape)
                # print(labels.shape)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # according to ViT paper, all fine-tuned tasks do gradient clipping at global norm 1.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                batch_loss.append(loss.item())
                # if len(batch_loss) > 0:
                #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
                #     logging.info(
                #         '(client {}. Local Training Epoch: {}\tBatch:{}/{}\tLoss: {:.6f}'.format(self.client_index,
                #                                                                                  epoch,
                #                                                                                  batch_idx,
                #                                                                                  len(self.train_local),
                #                                                                                  sum(epoch_loss) / len(
                #                                                                                      epoch_loss)))
        weights = self.model.head.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def _train_raw_data(self):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.train_local):
                # logging.info(images.shape)
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs, _ = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # according to ViT paper, all fine-tuned tasks do gradient clipping at global norm 1.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                batch_loss.append(loss.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    logging.info(
                        '(client {}. Local Training Epoch: {}\tBatch:{}/{}\tLoss: {:.6f}'.format(self.client_index,
                                                                                                 epoch,
                                                                                                 batch_idx,
                                                                                                 len(self.train_local),
                                                                                                 sum(epoch_loss) / len(
                                                                                                     epoch_loss)))
        weights = self.model.head.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def test(self):
        # train data
        train_tot_correct, train_num_sample, train_loss = self._infer(self.train_local)

        # test data
        test_tot_correct, test_num_sample, test_loss = self._infer(self.test_local)

        # test on training dataset
        train_acc = train_tot_correct / train_num_sample
        train_loss = train_loss / train_num_sample

        # test on test dataset
        test_acc = test_tot_correct / test_num_sample
        test_loss = test_loss / test_num_sample
        return train_acc, train_loss, test_acc, test_loss

    def _infer_on_raw_data(self, test_data):
        self.model.eval()
        self.model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                time_start_test_per_batch = time.time()
                x = x.to(self.device)
                target = target.to(self.device)
                pred, _ = self.model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
                time_end_test_per_batch = time.time()
                logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))

        return test_acc, test_total, test_loss

    def _infer(self, test_data):
        time_start_test_per_batch = time.time()
        self.model.eval()
        self.model.to(self.device)

        if test_data == self.train_local:
            test_data_extracted_features = self.train_data_extracted_features
        else:
            test_data_extracted_features = self.test_data_extracted_features

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx in test_data_extracted_features.keys():
                (x, labels) = test_data_extracted_features[batch_idx]
                x = x.to(self.device)
                labels = labels.to(self.device)

                log_probs = self.model.head(x)
                loss = criterion(log_probs, labels)
                _, predicted = torch.max(log_probs, -1)
                correct = predicted.eq(labels).sum()
                test_acc += correct.item()
                test_loss += loss.item() * labels.size(0)
                test_total += labels.size(0)
                time_end_test_per_batch = time.time()

        # logging.info("time per _infer = " + str(time_end_test_per_batch - time_start_test_per_batch))
        return test_acc, test_total, test_loss
