import numpy as np
import os
import pathlib
from pathlib import Path
from zipfile import ZipFile
import zipfile
import zlib
import shutil

"""
Run this script in the folder that you want to zip the files too
"""

#### INPUTS ####
chunk_size = 2 # number of files in zip folders
################

# stackoverflow link: https://stackoverflow.com/a/46267469
# create a zip file by excluding the path of directory
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


def zipit(dir_list, zip_name):
    zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
    for dir in dir_list:
        zipdir(dir, zipf)
    zipf.close()


def chunks(l, n):
    """Yield successive n-sized chunks from lst.
    from: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


# create list of files to chunk
file_array = []
for file in os.listdir(Path().absolute()):
    # only look at .pickle files
    if file.endswith("_encoder"):
        file_array.append(file)

file_array = list(chunks(file_array, chunk_size))

index_counter = 0
file_name = "{}.zip".format(str(index_counter))
for i in file_array:
    print(index_counter, i)
    zipit(i, file_name)
    index_counter += 1
    file_name = "{}.zip".format(str(index_counter))

# create the folder to move the zips into
parent_dir = Path.cwd().parents[0]
save_name = str(parent_dir).split('/')[-1]+'_zip'

pathlib.Path(parent_dir / save_name).mkdir(parents=True, exist_ok=True)

