import argparse
import logging
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.distributed.fed_transformer.utils import count_parameters
from fedml_api.model.cv.transformer.vit.vision_transformer import VisionTransformer, CONFIGS
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from fedml_api.distributed.fed_transformer.FedAvgAPI import FedML_init, FedML_Fed_Transformer_distributed


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='transformer', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")

    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--server_node_gpu_id', type=int, default=0,
                        help='server_node_gpu_id')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")

    parser.add_argument("--pretrained_dir", type=str,
                        default="./../../../fedml_api/model/cv/pretrained/Transformer/vit/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, args)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "transformer":
        model_type = 'vit-B_16'
        # pretrained on ImageNet (224x224), and fine-tuned on (384x384) high resolution.
        config = CONFIGS[model_type]
        logging.info("Vision Transformer Configuration: " + str(config))
        num_classes = output_dim
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
        # freeze the backbone
        for param in model.transformer.parameters():
            param.requires_grad = False

        model.load_from(np.load(args.pretrained_dir))
        num_params = count_parameters(model)
        logging.info("Vision Transformer Model Size = " + str(num_params))
    return model


def init_training_device(args, process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:" + str(args.server_node_gpu_id) if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device

def train():
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

def test():
    pass


def _extract_features(self):
    self.model.eval()
    self.model.to(self.device)

    path_train = "./extracted_features/" + str(self.client_index) + "-train.pkl"
    path_test = "./extracted_features/" + str(self.client_index) + "-test.pkl"
    train_data_extracted_features = dict()
    test_data_extracted_features = dict()
    if path.exists(path_train):
        train_data_extracted_features = load_from_pickle_file(path_train)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.train_local):
                time_start_test_per_batch = time.time()
                x = x.to(self.device)
                extracted_feature_x, _ = self.model.transformer(x)
                train_data_extracted_features[batch_idx] = (extracted_feature_x[:, 0].cpu().detach(), target)
                time_end_test_per_batch = time.time()
                logging.info("train_local feature extraction - time per batch = " + str(
                    time_end_test_per_batch - time_start_test_per_batch))
        save_as_pickle_file(path_train, train_data_extracted_features)

    if path.exists(path_test):
        test_data_extracted_features = load_from_pickle_file(path_test)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_local):
                time_start_test_per_batch = time.time()
                x = x.to(self.device)
                extracted_feature_x, _ = self.model.transformer(x)
                test_data_extracted_features[batch_idx] = (extracted_feature_x[:, 0].cpu().detach(), target)
                time_end_test_per_batch = time.time()
                logging.info("test_local feature extraction - time per batch = " + str(
                    time_end_test_per_batch - time_start_test_per_batch))
        save_as_pickle_file(path_test, test_data_extracted_features)

    return train_data_extracted_features, test_data_extracted_features

if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fed_transformer",
            name="FedTransformer(d)" + str(args.epochs) + "-lr" + str(args.lr),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    # logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(args, process_id, worker_number - 1, args.gpu_num_per_server)

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    # model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model = create_model(args, model_name=args.model, output_dim=10)

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

    self.train_data_extracted_features, self.test_data_extracted_features = self._extract_features()

    train()

    test()