
ckpt=/root/.cache/huggingface/hub/models--TencentARC--Mira-v0/snapshots/93c79a246c929e84e3a1dd9b044aa9cda880c337/384-v0.pt



HOST_GPU_NUM=1
HOST_NUM=1
CHIEF_IP=127.0.0.1
INDEX=0


torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=12594 --node_rank=$INDEX \
mira/scripts/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed 2000 \
--ckpt_path  $ckpt \
--base "./configs/Mira/config_384_mira.yaml" \
--savedir ./Saved_Text2Video-Test/test-mira-384 \
--n_samples 1 \
--bs 1 \
--height 240 \
--width 384 \
--unconditional_guidance_scale 12 \
--ddim_steps  50 --ddim_eta 1.0 \
--prompt_file ./prompts/test_prompt.txt
