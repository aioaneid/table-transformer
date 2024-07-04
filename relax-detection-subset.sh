d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Detection &&
    ks="1" &&
    nice ionice conda run --no-capture-output --live-stream -n tatr python relax/relax_subset.py --input_pascal_voc_filelist $d/train_filelist.txt --output_dir_root $d --seed 29722332 --shuf_filelist $d/shuf_train_filelist.txt --ks $ks
