gpu_id=0
dataset="yelp"
model_name="bert4rec"
noise_model="unet"
guide_model="cg"
guide_type="item"
aug_num=10
seed=42

python train_diffusion.py --gpu_id ${gpu_id} \
                        --dataset ${dataset} \
                        --noise_model ${noise_model} \
                        --guide_model ${guide_model} \
                        --guide_type ${guide_type} \
                        --num_workers 8 \
                        --hidden_size 64 \
                        --log \
                        --num_train_epochs 3000 \
                        --train \
                        --aug_num 10 \
                        --ema \
                        --check_path ${guide_model}_${noise_model} \
                        --seed ${seed} \
                        --guide

python train_diffusion.py --gpu_id ${gpu_id} \
                        --dataset ${dataset} \
                        --noise_model ${noise_model} \
                        --guide_model ${guide_model} \
                        --guide_type ${guide_type} \
                        --num_workers 8 \
                        --hidden_size 64 \
                        --log \
                        --pretrain_dir diffusion/${guide_model}_${noise_model} \
                        --aug_file aug_cg \
                        --aug_num 10 \
                        --seed ${seed} \
                        --guide

python train_diffusion.py --gpu_id ${gpu_id} \
                        --dataset ${dataset} \
                        --noise_model ${noise_model} \
                        --guide_model ${guide_model} \
                        --guide_type ${guide_type} \
                        --num_workers 8 \
                        --hidden_size 64 \
                        --log \
                        --num_train_epochs 3000 \
                        --train \
                        --aug_num 10 \
                        --ema \
                        --check_path ${guide_model}_${noise_model} \
                        --guide \
                        --seed ${seed}

python train_diffusion.py --gpu_id ${gpu_id} \
                        --dataset ${dataset} \
                        --noise_model ${noise_model} \
                        --guide_model ${guide_model} \
                        --guide_type ${guide_type} \
                        --num_workers 8 \
                        --hidden_size 64 \
                        --log \
                        --classifier_scale ${cs} \
                        --pretrain_dir diffusion/${guide_model}_${noise_model} \
                        --aug_file aug_cf \
                        --aug_num 10 \
                        --guide \
                        --seed ${seed}