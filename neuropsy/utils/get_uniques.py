import os
import re
from collections import OrderedDict


def get_subject_ids_from_path(path) -> list:
    regex = re.compile(r'^sub\d{2}', re.IGNORECASE)

    files = os.listdir(path)

    sub_ids = []
    for file in files:
        if file.startswith('sub'):
            sub_file = regex.findall(file)
            if sub_file != []:
                sub_ids.append(sub_file[0].split('.')[0][3:])
    sub_ids = list(OrderedDict.fromkeys(sub_ids))
    return sub_ids
