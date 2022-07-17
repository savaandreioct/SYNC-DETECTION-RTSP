"""
Module containing util methods used in project
"""

from typing import List


def load_file_as_list(file: str) -> List[str]:
    """
    Returns a list containing each line in the file as a list element
    """

    txt_file = open(file, "r")
    file_content = txt_file.read()
    content_list = file_content.split("\n")
    txt_file.close()
    return content_list
