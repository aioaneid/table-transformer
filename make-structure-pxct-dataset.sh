d=~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/6a764b276769537108c1f6250f7eaafc65f79078/PubTables-1M-Structure &&
    pxc="inf" &&
    if [[ ! -e ${d}/train_pxct_${pxc}_filelist.txt ]]; then
        mkdir -p ${d}/train_pxct_${pxc} &&
            cat ${d}/train_filelist.txt |
while read -r f; do
                if [[ -e ${d}/train_pxc_${pxc}/$(basename ${f}) ]]; then
                    replacement=$(cat ${d}/${f} | tr '\n' ' ' | grep -oP '<object>\s*<name>table</name>.*?</object>' | sed 's#<object>\s*<name>table</name>##') &&
                    cat ${d}/train_pxc_${pxc}/$(basename ${f}) | tr '\n' ' ' | perl -pe "s#(<object>\s*<name>table \w+ o</name>).*?</object>#\$1$replacement#" |
                    xmllint --format - >${d}/train_pxct_${pxc}/$(basename ${f})
                else
                    ln -sf ${d}/${f} ${d}/train_pxct_${pxc}/$(basename ${f})
                fi
            done &&
            sed "s#train/#train_pxct_${pxc}/#" ${d}/train_filelist.txt >${d}/train_pxct_${pxc}_filelist.txt
    fi
