#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from random import sample
from enum import Enum, auto
from typing import NamedTuple, Optional, Tuple
from time import time
from collections import deque, namedtuple, defaultdict
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from RL_Model import (
    AFParams, TaskType, SDT, RandomForestSDT,
    AttentionForest, MultiHeadAttentionABRF, Patient, Clinic, Department, Physician,
    TimeSlot, build_clinics, SchedulingEnv, PolicyNetwork, ValueNetwork, ReplayMemory, MPPPOAgent, train_mpppo
)

# In[ ]:


data_path = r'dataset_encoded_1.1.xlsx'  # private dataset
df = pd.read_excel(data_path)

Features = ['LANGUAGE', 'VISIT_TYPE', 'INSTITUTE',
    'CENTER_NAME', 'DEPARTMENT_NAME', 'PROVIDER',
    'GENDER', 'VISIT_REASON', 'DAY', 'MONTH',
    'TIME', 'AGE_GROUP', 'SEASON',
    'NUMBER_OF_VISIT', 'NUMBER_OF_APPOINTMENT_ON_THE_SAME_DAY',
    'WEATHER_CONDITIONS', 'AIR_QUALITY',
    'WEEK_OF_THE_MONTH', '%NO_SHOW', 'TEMPERATURE', 'DEW',
    'HUMIDITY', 'WINDSPEED', 'VISIBILITY']

# Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[Features]
y = df['APPT_STATUS']

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)  # In-place transformation on X



# In[ ]:

PATH = r"C:\Users\USER\DB-RL\MHASRF_full_1.0.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(PATH, map_location=device, weights_only=False)
model.to(device).eval()
  


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_vectors = [
[1.0, 0.0, 0.0],   
[0.0, 1.0, 0.0],   
[0.0, 0.0, 1.0],   
[0.5, 0.25, 0.25],
[0.25, 0.5, 0.25],
[0.25, 0.25, 0.5],
[0.33, 0.33, 0.34],
[0.7, 0.2, 0.1],   
[0.2, 0.7, 0.1],  
[0.2, 0.1, 0.7],   
]

clinics = build_clinics()

env = SchedulingEnv(
    clinics=clinics,
    df_shuffled=df_shuffled,
    Features=Features,
    scaler=scaler,
    model=model,
    booking_horizon=14,
    lambda_booking=100,
    seed=42,
    warmup_days=14,
)

trained_agents = train_mpppo(
    env,
    epochs=250,
    weight_vectors=weight_vectors,
    episodes_per_epoch=5,
    lr_policy=1e-3,   # float
    lr_value=1e-3     # float,
)


