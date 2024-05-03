

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
export NCCL_TOPO_FILE=/tmp/topo.txt





HOST_NUM=1
INDEX=$1
CHIEF_IP=127.0.0.1







HOST_GPU_NUM=1



# args
name="config_384_mira"
config_file="configs/Mira/config_384_mira.yaml"
save_root="/root/workspace/save_path/mira"

cd mira/scripts
mkdir -p $save_root/$name


# export http_proxy=http://localhost:8118
# export https_proxy=http://localhost:8118

torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=14073 --node_rank=$INDEX \
trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=$HOST_NUM \
--auto_resume \
