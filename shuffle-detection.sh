d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Detection &&
    for train_split_name in train; do
        if [[ ! -f ${d}/shuf_${train_split_name}_filelist.txt ]]; then
            shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:238462sdfa2 -nosalt -pbkdf2 -iter 1000000 </dev/zero 2>/dev/null) ${d}/${train_split_name}_filelist.txt >${d}/shuf_${train_split_name}_filelist.txt
        fi
    done
