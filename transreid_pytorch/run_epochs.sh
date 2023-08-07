pretrain_dir=$1
output_fix=$2 #vits.ibot.lup.5.256x128.wpt.csk.4-8.ar.375
epoch=${3:-240}
device=${4:-0}
hw_ratio=${5:-2}
arch=${6:-small}

epoch_str=`printf %04d ${epoch}`
if [ $epoch -eq 300 ]; then
    pretrain=${pretrain_dir}/checkpoint.pth
else
    pretrain=${pretrain_dir}/checkpoint${epoch_str}.pth
fi
output=${output_fix}.e${epoch_str}

cmd="sh run_batch.sh $pretrain $output $device $hw_ratio $arch"
echo start to $cmd
$cmd
echo end to $cmd
