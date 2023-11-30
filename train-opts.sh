#!/bin/sh

while getopts " -:" OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
   case "$OPT" in
      train_xml_fileset )
         train_xml_fileset="$OPTARG";;
      train_split_name )
         train_split_name="$OPTARG";;
      train_max_size )
         train_max_size="$OPTARG";;
      matcher )
         matcher="$OPTARG";;
      batch_size )
         batch_size="$OPTARG";;
      reuse )
         reuse="$OPTARG";;
      eb )
         eb="$OPTARG";;
      seed )
         seed="$OPTARG";;
      epoch_seeds )
         epoch_seeds="$OPTARG";;
      torch_printoptions )
         torch_printoptions="$OPTARG";;
      debug_dataset )
         debug_dataset="$OPTARG";;
      debug_engine )
         debug_engine="$OPTARG";;
      epochs )
         typeset -i epochs="$OPTARG";;
     \? | *) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

echo "train_xml_fileset: $train_xml_fileset"
echo "train_split_name: $train_split_name"
echo "train_max_size: $train_max_size"
echo "matcher: $matcher"
echo "batch_size: $batch_size"
echo "reuse: $reuse"
echo "eb: $eb"
echo "seed: $seed"
echo "epoch_seeds: $epoch_seeds"
echo "torch_printoptions: $torch_printoptions"
echo "debug_dataset: $debug_dataset"
echo "debug_engine: $debug_engine"
echo "epochs: $epochs"

device=cpu && val_max_size="$(($train_max_size*10))" && train_instant=202312240000000000 && code=table-transformer && model_save_dir=/data/models/structure/pubmed/code/${code}/matcher/${matcher}/eb/${eb}/train_xml_fileset/${train_xml_fileset}/train_split_name/${train_split_name}/train_max_size/$train_max_size/batch_size/${batch_size}/output/train_instant/${train_instant} && d=/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/6a764b276769537108c1f6250f7eaafc65f79078/PubTables-1M-Structure && if test -f "/tmp/train-opts.stop"; then rm "/tmp/train-opts.stop";  else log_filename=train_pubmed_code_${code}_matcher_${matcher}_eb_${eb}_train_xml_fileset_${train_xml_fileset}_train_split_name_${train_split_name}_train_max_size_${train_max_size}_batch_size_${batch_size}_train_instant_${train_instant}_epochs_${epochs}_structure_main.log && (HUNGARIAN_MATCH_ALGO="$matcher" nice ionice python src/main.py --seed ${seed} --torch_printoptions ${torch_printoptions} --data_type structure --config_file src/structure_config.json --data_root_dirs $d --data_root_image_extensions .jpg --data_root_multiplicities 1 --device $device --mode train --train_xml_fileset "${train_xml_fileset}" --train_split_name ${train_split_name} --batch_size ${batch_size} --model_save_dir "$model_save_dir" --epochs $epochs --epoch_seeds ${epoch_seeds} --coco_eval_prefix "/.cache/coco_eval/structure/pubmed/code/${code}/matcher/${matcher}/eb/${eb}/train_xml_fileset/${train_xml_fileset}/train_split_name/${train_split_name}/train_max_size/$train_max_size/train_instant/${train_instant}/train_validation" --train_max_size ${train_max_size} --val_max_size ${val_max_size} --${eb} --${debug_dataset} --${debug_engine} $(if test $reuse -eq "1"; then echo "--model_load_path ${model_save_dir}/model.pth"; else echo ""; fi) |& tee ~/work/tmp/$log_filename); fi