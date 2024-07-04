d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Detection &&
    for test_split_name in train val test; do
        if [[ ! -f ${d}/${test_split_name}_single_filelist.txt ]]; then
            mkdir -p ${d}/${test_split_name}_single &&
                grep -o -c '<object>' -R ${d}/${test_split_name} | awk -F: '{if ($2 == 1){print $1}}' |
                while read -r f; do
                    ln -sf $f ${f/${test_split_name}\//${test_split_name}_single\/}
                done &&
                sed "s#${test_split_name}/#${test_split_name}_single/#" ${d}/${test_split_name}_filelist.txt >/tmp/original_order_filelist_for_single.txt &&
                ls -U ${d}/${test_split_name}_single | sed "s#.\+#${test_split_name}_single/\0#" >/tmp/existing_filelist_for_single.txt &&
                cat /tmp/original_order_filelist_for_single.txt |
                while read line; do
                    grep $line /tmp/existing_filelist_for_single.txt
                done >${d}/${test_split_name}_single_filelist.txt
        fi
    done
