#Single GPU
python train.py --config_file configs/market/vit_small_ics.yml

python train.py --config_file configs/msmt17/vit_small.yml DATASETS.ROOT_DIR ./data/ SOLVER.BASE_LR 4e-4 MODEL.DEVICE_ID '("1")' MODEL.PRETRAIN_PATH ../../ibot/log/vits.msmt17.224x224/checkpoint.pth OUTPUT_DIR logs/vits.ibot.msmt17.224x224.e300 MODEL.PRETRAIN_HW_RATIO 1 MODEL.POOLING 'avg_max'

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/market/vit_small_ics_ddp.yml

# test batch
sh -x run_batch.sh ../ibot/pretrained/vits16.ibot.teacher.pth vits16.ibot.inet.pt 3 1

# test epochs
sh run_epochs.sh  ../ibot/log/vits.lup.5.256x128.wpt.csk.4-8.ar.375/ vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375 200 0
sh run_epochs.sh  ../ibot/log/vits.lup.5.256x128.wpt.csk.4-8.ar.375/ vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375 240 0
sh run_epochs.sh  ../ibot/log/vits.lup.5.256x128.wpt.csk.4-8.ar.375/ vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375 300 2
sh run_epochs.sh  ../ibot/log/vitb.lup.03.256x128.wpt.csk.4-8.ar.375/ vitb.ibot.lup.03.256x128.wpt.csk.4-8.ar.375 240 1 2 base
