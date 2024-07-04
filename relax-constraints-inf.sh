code=$(basename $PWD) &&
    d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/6a764b276769537108c1f6250f7eaafc65f79078/PubTables-1M-Structure &&
    px='inf' &&
    seconds_since_epoch="$(date +%s)" &&
    find ${d}/train ! -name 'PMC4403077_table_0.xml' -type f -print0 | shuf -z |
    xargs -0 nice -n "$(($RANDOM % 20 + 1))" ionice -c 3 conda run --no-capture-output --live-stream -n tatr OPENBLAS_NUM_THREADS=1 python relax/relax_constraints.py --stop_file_path /tmp/relax-constraints-inf.stop --stop_file_wait_seconds 600 --input_words_data_dir ${d}/words --output_pascal_voc_xml_dir ${d}/train_pxc_${px} --centered_hole_side_constant=-${px} --centered_outer_side_constant ${px} --algo random_sequential --random_sequential_step $(
        for ((p = 11; p--; )); do
            echo $((1 << p))
        done
    ) --random_sequential_repeats 2 --skip_if_output_exists --post_checks --input_pascal_voc_xml_files |& tee -a ~/work/tmp/relax_constraints_code_${code}_px_${px}_seconds_since_epoch_${seconds_since_epoch}.log
