import copy
import logging
import time

import torch
import wandb
import numpy as np
from torch import nn

from fedml_api.distributed.fed_transformer.utils import transform_list_to_tensor


class FedAVGAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, model, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model = model

        self.train_acc_client_dict = dict()
        self.train_loss_client_dict = dict()
        self.test_acc_client_dict = dict()
        self.test_loss_client_dict = dict()

    def get_global_model_params(self):
        return self.model.head.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.model.head.load_state_dict(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def add_client_test_result(self, client_index, train_acc, train_loss, test_acc, test_loss):
        logging.info("################add_client_test_result : {}".format(client_index))
        self.train_acc_client_dict[client_index] = train_acc
        self.train_loss_client_dict[client_index] = train_loss
        self.test_acc_client_dict[client_index] = test_acc
        self.test_loss_client_dict[client_index] = test_loss

    def output_global_acc_and_loss(self, round_idx):
        logging.info("################output_global_acc_and_loss : {}".format(round_idx))

        # test on training dataset
        train_acc = np.array([self.train_acc_client_dict[k] for k in self.train_acc_client_dict.keys()]).mean()
        train_loss = np.array([self.train_loss_client_dict[k] for k in self.train_loss_client_dict.keys()]).mean()
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        logging.info(stats)

        # test on test dataset
        test_acc = np.array([self.test_acc_client_dict[k] for k in self.test_acc_client_dict.keys()]).mean()
        test_loss = np.array([self.test_loss_client_dict[k] for k in self.test_loss_client_dict.keys()]).mean()
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats)