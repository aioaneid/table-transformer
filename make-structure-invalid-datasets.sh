padding=_PADDING_2 &&
    d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/6a764b276769537108c1f6250f7eaafc65f79078/PubTables-1M-Structure${padding} &&
    for test_split_name in val test; do
        comm -3 <(sed "s#${test_split_name}_pxc_0/#${test_split_name}/#" ${d}/${test_split_name}_pxc_0_filelist.txt | sort) <(sort ${d}/${test_split_name}_filelist.txt) | sed -e 's/[[:space:]]*//' -e 's#/#_invalid/##' >${d}/${test_split_name}_invalid_filelist.txt &&
            mkdir -p ${d}/${test_split_name}_invalid &&
            cat ${d}/${test_split_name}_invalid_filelist.txt | while read -r f; do
                ln -s ${d}/${test_split_name}/$(basename ${f}) ${d}/${f}
            done
    done
