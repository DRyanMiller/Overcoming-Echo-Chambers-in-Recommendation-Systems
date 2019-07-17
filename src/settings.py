# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def set_env():
    """ Simple main() to initialize environment variables.
    """
    print("Loading environment variables")

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

if __name__ == '__set_env__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]  
    print("Executed when invoked directly")

    set_env()
else:
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
 
    #print("Executed when imported")
    
    set_env()
