"""
Created on Fri Mar  5 10:42:58 2021

@author: Oyelade
"""
from numpy import concatenate, savetxt, array
from csv import DictWriter
from os import getcwd, path, makedirs
from pandas import read_csv 

def save_results_to_csv(item=None, filename=None, pathsave=None):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
    with open(check_directory + filename + ".csv", 'a') as file:
        w = DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=item.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(item)

def save_solutions_to_csv(solutions=None, filename=None, pathsave=None):
    savetxt(pathsave + filename + ".csv", array(solutions), delimiter=",")
    return None

def check_make_dir(pathsave=None):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
 