# -*- coding: utf-8 -*-
## base.py
## wrapper for all local imports
from dotenv import find_dotenv, load_dotenv

def test_base():
    print("Base Module Imported")
    print("\nTesting local imports")
   

    return None

def getenv_variables():
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


getenv_variables()