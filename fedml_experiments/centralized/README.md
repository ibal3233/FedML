

# PyTorch DDP demo
# sh run_ddp.sh 8 2 0 192.168.11.1 11111
nohup sh run_ddp.sh 8 2 0 192.168.11.1 11111 > ./machine1.txt 2>&1 &
nohup sh run_ddp.sh 8 2 1 192.168.11.1 11111 > ./machine2.txt 2>&1 &