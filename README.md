# TATR with Box Relaxation

Clone of <https://github.com/microsoft/table-transformer> with changes to enable box relaxation.

Usage:

```sh
conda env create --name tatr --file=environment.yml

conda run --no-capture-output --live-stream -n tatr python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="bsmock/pubtables-1m", repo_type="dataset")'
```

Instead of [environment.yml](./environment.yml) it is also possible to use [environment-latest](./environment-latest.yml).

The available list of scripts is described below. Note that only those using python need conda, and even those can be easily modified to skip conda if one installs manually the list of dependencies.

- Table Detection (TD)
  - [make-single-object-dataset.sh](./make-single-object-dataset.sh): Makes a training dataset out of the single-table images;
  - [shuffle-detection.sh](./shuffle-detection.sh): Shuffles the list of training images. Prerequisite for [relax-detection-only-subset.sh](./relax-detection-only-subset.sh) and [relax-detection-subset.sh](./relax-detection-subset.sh);
  - [relax-detection-only-subset.sh](./relax-detection-only-subset.sh): Keeps a (randomly selected) table from each training image;
  - [relax-detection-subset.sh](./relax-detection-subset.sh): Keeps a (randomly selected) table from each training image and relaxes the other ones to have no hole border and the full image as the outer hole.
  - [relax-dataset-detection-px.sh](./relax-dataset-detection-px.sh): Contracts (for the hole border) and expands (for the outer border) each training table by 2 pixels. Each of the 4 sides may be independently contracted less and/or expanded less than 2 pixels if needed to keep the table center within the hole border resp. to keep the outer border within the image box.
  - [relax-dataset-detection-pxs.sh](./relax-dataset-detection-pxs.sh): Contracts (for the hole border) and expands (for the outer border) each training table by 4 pixels for one dataset, and once again by 8 pixels for yet another training dataset. The actual amount of relaxation on each side can be lower if necessary to maintain symmetry between corresponding edges of the hole and outer borders.
  - [train-detection-px.sh](./train-detection-px.sh): Trains the model with the dataset created by [relax-dataset-detection-px.sh](./relax-dataset-detection-px.sh);
  - [train-detection-midline.sh](./train-detection-midline.sh): Trains the model with the dataset created by [relax-dataset-detection-pxs.sh](./relax-dataset-detection-pxs.sh);
- Table Structure Recognition (TSR)
  - [relax-constraints-inf.sh](./relax-constraints-inf.sh): Relax the TSR objects in the training dataset while making sure that the same table cell matrix ensues. Can be run in parallel, e.g. one process pe CPU core. Upon restart continues where it left over, with the exception that the tables with at least a spanning cell which is dropped during matrix cell extraction are reprocesses after each restart.
  - [make-structure-pxct-dataset.sh](./make-structure-pxct-dataset): Makes the table outer border identical to the original table bounding box. Makes a complete training dataset, including without relaxation those tables with spanning cells which get dropped in the cell matrix extraction step.
  - [train-structure-pxct.sh](./train-structure-pxct.sh): Trains a TSR model on the constrained box relaxation dataset.

The GriTS evaluation code can be executed in parallel on different batches of images, e.g.:

```sh
seed=$((echo 0 ${test_split_name} ${epoch} | sha512sum | awk '{printf "ibase=16; "toupper($1)}' && echo " % 7FFFFF") | bc) &&
conda run --no-capture-output --live-stream -n tatr python src/main.py --data_type structure --config_file src/structure_config.json --data_root_dirs ${d} --table_words_dir ${d}/words --data_root_image_extensions .jpg --data_root_multiplicities 1 --device ${device} --mode ${mode} --test_split_name ${test_split_name} --test_start_offset ${test_start_offset} --test_max_size ${test_max_size} --no-enable_bounds --model_load_path ${f}/model_${epoch}.pth --metrics_save_filepath ${metrics_save_path} --seed ${seed} --torch_num_threads 1
```

The metrics batches for an arbitrary epoch can then be merged together using [plots/aggregate_json_grits.py](./plots/aggregate_json_grits.py).

Note that the training scripts allow a new `--mode` option `validate` which can be executed in a subsequent phase to training.
