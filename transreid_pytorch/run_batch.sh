pretrain=$1
output_fix=$2 #vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375
device=${3:-0}
hw_ratio=${4:-2}
arch=${5:-small}
python train.py --config_file configs/msmt17/vit_${arch}.yml DATASETS.ROOT_DIR ./data/ SOLVER.BASE_LR 4e-4 MODEL.DEVICE_ID '("'${device}'")' MODEL.PRETRAIN_PATH ${pretrain} OUTPUT_DIR logs/msmt.${output_fix} MODEL.PRETRAIN_HW_RATIO ${hw_ratio}
python train.py --config_file configs/market/vit_${arch}.yml DATASETS.ROOT_DIR ./data/ SOLVER.BASE_LR 4e-4 MODEL.DEVICE_ID '("'${device}'")' MODEL.PRETRAIN_PATH ${pretrain} OUTPUT_DIR logs/market.${output_fix} MODEL.PRETRAIN_HW_RATIO ${hw_ratio}
python train.py --config_file configs/dukemtmc/vit_${arch}.yml DATASETS.ROOT_DIR ./data/ SOLVER.BASE_LR 4e-4 MODEL.DEVICE_ID '("'${device}'")' MODEL.PRETRAIN_PATH ${pretrain} OUTPUT_DIR logs/duke.${output_fix} MODEL.PRETRAIN_HW_RATIO ${hw_ratio}
python train.py --config_file configs/occ_duke/vit_${arch}.yml DATASETS.ROOT_DIR ./data/ SOLVER.BASE_LR 4e-4 MODEL.DEVICE_ID '("'${device}'")' MODEL.PRETRAIN_PATH ${pretrain} OUTPUT_DIR logs/occ_duke.${output_fix} MODEL.PRETRAIN_HW_RATIO ${hw_ratio}
