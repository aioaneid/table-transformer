#!/bin/sh

device=cuda &&
train_instant=202403080000000000 &&
code=$(basename $(pwd)) &&
val_max_size=0 &&
d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Detection &&
mkdir -p ~/work/tmp/detection/train &&
for epochs in 1 2 4 8 16 20; do
    for px in 4 8 16; do
        if [[ -f "/tmp/train-detection-main-${px}.stop" ]]; then
            echo "Skipping epochs: ${epochs} px: ${px}"
        else
            eb="enable_bounds" &&
            model_save_dir=/data/models/detection/pubmed/code/${code}/px/${px}/eb/${eb}/output/train_instant/${train_instant} &&
            log_filename=train_pubmed_code_${code}_px_${px}_eb_${eb}_train_instant_${train_instant}_epochs_${epochs}_detection_main.log &&
            wait_for_cuda_train &&
            conda run --no-capture-output --live-stream -n ttt python src/main.py --data_type detection --config_file src/detection_config.json --data_root_dirs ${d} --data_root_image_extensions .jpg --data_root_multiplicities 1 --device ${device} --mode train --train_split_name train_px_${px} --val_max_size ${val_max_size} --model_save_dir ${model_save_dir} --epochs ${epochs} --${eb} $(if test -f ${model_save_dir}/model.pth; then echo "--model_load_path ${model_save_dir}/model.pth"; else echo ""; fi) |& tee -a ~/work/tmp/detection/train/${log_filename}
        fi
    done
done
