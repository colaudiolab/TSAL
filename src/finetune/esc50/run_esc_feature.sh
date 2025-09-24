pretrain_exp=
pretrain_model=SSAST-Base-Patch-400
pretrain_path=./${pretrain_exp}/${pretrain_model}.pth
dataset=esc50
dataset_mean=-6.6268077
dataset_std=5.358466
target_length=512
noise=True
bal=none
lr=1e-4
freqm=24
timem=96
mixup=0
epoch=50
batch_size=32
fshape=16
tshape=4
fstride=16
tstride=4
sampling='margin'
task=ft_avgtok
model_size=base
head_lr=1
cycles=7
folds=1
for((fold=1;fold<=$folds;fold++));
do
  exp_dir=${dataset}/${sampling}/${fold}
  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json
  CUDA_VISIBLE_DEVICES=0 python -W ignore ../../run.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --sampling ${sampling} \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
  --model_size ${model_size} --adaptschedule False \
  --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --cycles $cycles --fold $fold --folds $folds \
  --lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc
done
