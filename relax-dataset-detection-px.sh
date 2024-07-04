#!/bin/sh

d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Detection &&
    for px in 2; do
        if [[ ! -f "$d/train_px_${px}_filelist.txt" ]]; then
            (
                find $d/train -type f -print0 |
                    xargs -0 nice -n 19 ionice -c 3 conda run --no-capture-output --live-stream -n tatr python relax/relax_dataset.py --output_pascal_voc_xml_dir $d/train_px_${px} --inner_multiplier 1 1 1 1 --inner_constant " -${px}" " -${px}" " -${px}" " -${px}" --outer_multiplier 1 1 1 1 --outer_constant ${px} ${px} ${px} ${px} --clamp_outer_boundary image --enclose_inner_boundary none --input_pascal_voc_xml_files
            ) &&
                sed "s#train/#train_px_${px}/#" $d/train_filelist.txt >${d}/train_px_${px}_filelist.txt
        fi
    done
