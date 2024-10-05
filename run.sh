source /private/home/wangru/wr_env.conf

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=False

log_time=$(date "+%Y%m%d%H%M%S")

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

nohup deepspeed --num_gpus=8 main.py --train_args_file config/pretrain/zero-pretrain-full.yaml &> log/zero_pt_${log_time}.log &
