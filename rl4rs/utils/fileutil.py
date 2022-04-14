import os
import sys
import glob
import numpy as np


def find_match_files(pattern, search_path, pathsep=os.pathsep):
    for path in search_path.split(pathsep):
        for match in glob.glob(os.path.join(path, pattern)):
            yield match


def find_newest_files(pattern, search_path, pathsep=os.pathsep):
    files = []
    timestamps = []
    for path in search_path.split(pathsep):
        for match in glob.glob(os.path.join(path, pattern)):
            files.append(match)
            timestamps.append(float(os.path.getctime(match)))
    if len(files) > 0:
        return files[np.argmax(timestamps)]
    else:
        return ''
