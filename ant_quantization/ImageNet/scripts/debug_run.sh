mkdir -p log

CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --dataset=imagenet --model=resnet18 --mode=int --wbit=4 --abit=4 --w_low 1 --a_low 1 \
--calib_size=1024 --ptq --w_opt_target tensor --w_opt_metric mse --a_opt_target tensor --a_opt_metric mse > ./log/debug_resnet18_tensor_mse.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46667 main.py --dataset=imagenet --model=resnet18 --mode=int --wbit=4 --abit=4 --w_low 1 --a_low 1 \
--calib_size=1024 --ptq --w_opt_target output --w_opt_metric mse --a_opt_target output --a_opt_metric mse > ./log/debug_resnet18_output_mse.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46668 main.py --dataset=imagenet --model=resnet18 --mode=int --wbit=4 --abit=4 --w_low 1 --a_low 1 \
--calib_size=1024 --ptq --w_opt_target output --w_opt_metric fisher_diag --a_opt_target output --a_opt_metric fisher_diag > ./log/debug_resnet18_output_fisher.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46669 main.py --dataset=imagenet --model=resnet18 --mode=int --wbit=4 --abit=4 --w_low 1 --a_low 1 \
--calib_size=1024 --ptq --w_opt_target activated_output --w_opt_metric mse --a_opt_target activated_output --a_opt_metric mse > ./log/debug_resnet18_aoutput_mse.log 2>&1 &