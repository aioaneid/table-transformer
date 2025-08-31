padding=_PADDING_2 &&
    code=$(basename $PWD) &&
    d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/6a764b276769537108c1f6250f7eaafc65f79078/PubTables-1M-Structure${padding} &&
    px='inf' &&
    dr='dryrun' &&
    test_split_name=val_invalid &&
    f=${d}/${test_split_name}/PMC2923187_table_2.xml &&
    conda run --no-capture-output --live-stream -n tatr OPENBLAS_NUM_THREADS=1 python relax/relax_constraints.py --${dr} --stop_file_path /tmp/relax-constraints-inf-dryrun.stop --stop_file_wait_seconds 600 --input_words_data_dir ${d}/words --centered_hole_side_constant=-${px} --centered_outer_side_constant ${px} --no-include_token_iob_threshold_constraints --no-include_token_max_iob_ordering_constraints --no-include_relative_iob_1d_constraints --no-include_relative_iob_2d_constraints --no-optimize --check_original_consistency --input_pascal_voc_xml_files $f
