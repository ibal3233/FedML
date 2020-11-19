import logging

import torch
from torch import nn

from fedml_api.distributed.fed_transformer.utils import transform_tensor_to_list, WarmupCosineSchedule, \
    WarmupLinearSchedule


class FedAVGTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model,
                 args):
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

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

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self):
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
        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number
