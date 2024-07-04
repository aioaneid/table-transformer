#!/bin/sh

device=cuda &&
train_instant=202403080000000000 &&
code=$(basename $(pwd)) &&
val_max_size=0 &&
d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/6a764b276769537108c1f6250f7eaafc65f79078/PubTables-1M-Structure &&
for epochs in 1 2 4 8 16 20 28; do
    for pxct in inf; do
        if test ! -f "/tmp/train-structure-pxct-${pxct}.stop"; then
            eb=$(if test "${pxct}" = "0"; then echo "no-enable_bounds"; else echo "enable_bounds"; fi) &&
            model_save_dir=/data/models/structure/pubmed/code/${code}/pxct/${pxct}/eb/${eb}/output/train_instant/${train_instant} &&
            log_filename=train_pubmed_code_${code}_pxct_${pxct}_eb_${eb}_train_instant_${train_instant}_epochs_${epochs}_structure_main.log &&
            conda run --no-capture-output --live-stream -n tatr python src/main.py --data_type structure --config_file src/structure_config.json --data_root_dirs ${d} --data_root_image_extensions .jpg --data_root_multiplicities 1 --device ${device} --mode train --train_split_name train_pxct_${pxct} --val_max_size ${val_max_size} --model_save_dir ${model_save_dir} --epochs ${epochs} --${eb} $(
                if test -f ${model_save_dir}/model.pth; then echo "--model_load_path ${model_save_dir}/model.pth"; else echo ""; fi
                ) |& tee -a ~/work/tmp/structure/train/${log_filename}
        fi
    done
done

