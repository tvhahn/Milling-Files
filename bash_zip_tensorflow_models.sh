#!/bin/bash
# bash file that will zip up all the tensorflow files
# directory of the results we want to zip
dir=/home/tim/Documents/milling/src/models/results_archive/2020.04.22_results_1/20200421-125428/

# Final name for saving zip files in
final_name=2020.04.22_results_1_zip

python_file=zip_tensorflow_models.py


cp "$python_file" "$dir/$python_file"
cd "$dir"

python "$python_file"
parentdir="$(dirname "$dir")"
savedir="$parentdir/$final_name"

## move all .zip files cdin interim_data_sample to main directory
mv *.zip "$savedir"