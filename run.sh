#!/bin/sh
pkill nmon
pkill nvidia-smi
pkill python3

python_path=/home/ubuntu/venv/bin/python3
src_path=/home/ubuntu/src/shell5/
record_path=/home/ubuntu/data/out-0
raw_path=/home/ubuntu/imagenet/
result_path=/home/ubuntu/result5/


mkdir result
mkdir result/alexnet_cpu_decode


num_gpu=8
num_epoch=2
batch_size=512

for num_worker in 4 5 6 7 8 9 10 11 12 13
do
for arch in alexnet resnet18 shufflenet_v2_x1_0
do
nohup nvidia-smi dmon -s pucvmet -c 500 -f result/alexnet_cpu_decode/${arch}_${num_worker}.gpu &
/home/ubuntu/nmon -s 1 -c 500 -F result/alexnet_cpu_decode/${arch}_${num_worker}.nmon
#sudo sync &&  echo 3 > /proc/sys/vm/drop_caches
$python_path -m torch.distributed.launch --nproc_per_node=$num_gpu ${src_path}dali_tfr_cpu.py --epochs $num_epoch  --workers $num_worker   -a $arch -b $batch_size --fp16  $record_path > result/alexnet_cpu_decode/${arch}_${num_worker}.data
pkill python3
pkill nmon
pkill nvidia-smi
sleep 1
done
done














