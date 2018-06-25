import os.path
import tarfile

"""
This module is only used to create the tar.gz files from folders containing the cell images.

We created tar files to make it easier to distribute the dataset. 
"""


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# convert the image folder into .tar.gz files for easier GitHub upload/download
# make_tarfile('./data/images/EOSINOPHIL.tar.gz', './data/images/EOSINOPHIL')
# make_tarfile('./data/images/LYMPHOCYTE.tar.gz', './data/images/LYMPHOCYTE')
# make_tarfile('./data/images/MONOCYTE.tar.gz', './data/images/MONOCYTE')
# make_tarfile('./data/images/NEUTROPHIL.tar.gz', './data/images/NEUTROPHIL')
