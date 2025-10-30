# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import sys
import subprocess
from pathlib import Path

#local import
from dataacquisition import dataacquisition

def install():
    req = Path("requirements.txt")
    if not req.exists():
        print("No requirements.txt found")
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])

def main():
    #install()
    dataacquisition("kaggle")

if __name__ == "__main__":
    main()