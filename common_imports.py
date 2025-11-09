# ======================================================
# Origin data download
import FinanceDataReader as fdr
import pandas_datareader.data as pdr
import yfinance as yf

# ======================================================
# TA-Lib
import talib

# ======================================================
# basic library
import argparse
import warnings

import pandas as pd
import numpy as np

import time
import math
import os
import os.path
import random
import shutil
import glob

from tqdm import tqdm
from datetime import datetime

import copy
from pathlib import Path

# ======================================================
# visualize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# ======================================================
# Managing files
import openpyxl
import pickle

# ======================================================
# Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score

# ======================================================
# torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm

from torch.cuda.amp import GradScaler, autocast

# ======================================================
# gmm preprocessing
from scipy.optimize import fmin
from scipy.special import lambertw
from scipy.stats import kurtosis, norm
import statsmodels.api as sm

# ======================================================
# metric
import numpy.linalg as linalg

from scipy import stats
from scipy.stats import ks_2samp
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.dates as mdates

plt.rcParams['savefig.dpi'] = 300  # dpi