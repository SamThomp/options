
from decimal import Decimal
import pyotp
import robin_stocks.robinhood as rh

import json

import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import ops

from datetime import datetime
import datetime as dt
import time as t

import numpy as np


# Deep learning parameters
SPLIT_FRACTION_TRAIN = 0.715
SPLIT_FRACTION_PREDICT = 0.5
STEP = 1
PAST = 64
FUTURE = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50

# Custom Parameters
PREV_HIST = 20
LOOKBACK_COLS = -1 * (PREV_HIST + 1 + 15)
