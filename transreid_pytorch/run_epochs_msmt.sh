pretrain_dir=$1
output_fix=$2 #vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375
epochs="200 220 280 300"
hw_ratio=${3:-2}
arch=${4:-small}

i=0
for epoch in $epochs
do
    
    if [ $epoch -eq 300 ]; then
        pretrain=${pretrain_dir}/checkpoint.pth
    else
        pretrain=${pretrain_dir}/checkpoint0${epoch}.pth
    fi
    output=${output_fix}.e${epoch}
    device=$i
    python train.py --config_file configs/msmt17/vit_${arch}.yml DATASETS.ROOT_DIR ./data/ SOLVER.BASE_LR 4e-4 MODEL.DEVICE_ID '("'${device}'")' MODEL.PRETRAIN_PATH ${pretrain} OUTPUT_DIR logs/msmt.${output} MODEL.PRETRAIN_HW_RATIO ${hw_ratio} &

    # i++
    i=$(($i+1))
done

wait
