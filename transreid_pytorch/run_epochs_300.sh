#! /bin/bash

pretrain_dir=$1
output_fix=$2 #vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375
epoch=${3:-240}
hw_ratio=${4:-2}
arch=${5:-small}

step=20
end=301
for ((i=$step; i < $end; i+=$step*4))
do
    i_1=$[i+step*1]
    i_2=$[i+step*2]
    i_3=$[i+step*3]
    echo $i, $i_1, $i_2, $i_3
    if [ $i -lt $end ]; then
        bash run_epochs.sh $pretrain_dir $output_fix $i 0 $hw_ratio $arch &
    fi

    if [ $i_1 -lt $end ]; then
        bash run_epochs.sh $pretrain_dir $output_fix $i_1 1 $hw_ratio $arch &
    fi

    if [ $i_2 -lt $end ]; then
        bash run_epochs.sh $pretrain_dir $output_fix $i_2 2 $hw_ratio $arch &
    fi

    if [ $i_3 -lt $end ]; then
        bash run_epochs.sh $pretrain_dir $output_fix $i_3 3 $hw_ratio $arch &
    fi

    wait
done
