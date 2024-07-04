d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Detection &&
    if [[ ! -f ${d}/train_only_filelist.txt ]]; then
        ks="1" &&
            nice ionice conda run --no-capture-output --live-stream -n tatr python relax/relax_subset.py --input_pascal_voc_filelist $d/train_filelist.txt --output_dir_root $d --seed 29722332 --ks $ks --remove_non_selected &&
            mv ${d}/train_head_2147483647_only_1 ${d}/train_only &&
            sed 's#train/#train_only/#' ${d}/train_filelist.txt >${d}/train_only_filelist.txt
    fi
