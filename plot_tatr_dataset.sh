#!/bin/sh

d=~/tmp/PubTables-1M-Structure_PADDING_2 &&
    OPENBLAS_NUM_THREADS=1 conda run --no-capture-output --live-stream -n tatr python plots/plot_tatr_image.py --input_structure_pascal_voc_xml ${d}/val/PMC2923187_table_2.xml --input_image_dir ${d}/images --background_image_extension '.png' --input_words_data_dir ${d}/words --output_image_dir ${d}/val-tatr-images
