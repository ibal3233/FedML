## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Download the pretrained Transformer weights
```
cd ./../../../fedml_api/model/cv/pretrained/Transformer/vit/
sh download_pretrained_weights.sh
cd ./../../../fedml_experiment/
```

## Experimental Tracking
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

wandb.init(project="fed_transformer")

## Run Experiments

## ResNet56 Federated Training

 CIFAR10
train on IID dataset 
```
sh run_fed_transformer.sh 10 10 1 4 transformer homo 100 1 64 0.01 cifar10 "./../../../data/cifar10" 0 0

sh run_fed_transformer.sh 8 8 1 8 transformer homo 100 20 64 0.003 cifar10 "./../../../data/cifar10" 0 0 

##run on background
nohup sh run_fed_transformer.sh 8 8 1 8 transformer homo 100 20 64 0.003 cifar10 "./../../../data/cifar10" 0 > ./fed-transformer-homo-cifar10.txt 2>&1 &

## Chaoyang's machine
nohup sh run_fed_transformer.sh 10 10 1 4 transformer homo 100 20 64 0.003 cifar10 "./../../../data/cifar10" 0 > ./fed-transformer-homo-cifar10.txt 2>&1 &

```



train on non-IID dataset
```
sh run_fed_transformer.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0 0

##run on background
nohup sh run_fed_transformer.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0 0 > ./fedavg-resnet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 0 > ./fedavg-resnet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 0 > ./fedavg-resnet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 0 > ./fedavg-resnet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 0 > ./fedavg-resnet-hetero-cinic10.txt 2>&1 &
```
