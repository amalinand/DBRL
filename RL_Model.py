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
from scipy.special import softmax as _softmax


# In[ ]:


class SDT(nn.Module):
    def __init__(self, input_dim, output_dim, depth=3, lamda=1e-3, use_cuda=True):
        super(SDT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Penalty per layer
        self.penalty_list = [self.lamda * (2 ** (-d)) for d in range(self.depth)]

        # Layers
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )
        self.leaf_nodes = nn.Linear(self.leaf_node_num_, self.output_dim, bias=False)

    def forward(self, X, feature_subset=None, is_training_data=False):
        if feature_subset is not None:
            X = X[:, feature_subset]
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)
        return (y_pred, _penalty) if is_training_data else y_pred

    def _forward(self, X):
        batch_size = X.size(0)
        X = self._data_augment(X)
        path_prob = self.inner_nodes(X).unsqueeze(2)
        path_prob = torch.cat([path_prob, 1 - path_prob], dim=2)

        _mu = X.new_ones(batch_size, 1, 1)
        _penalty = torch.tensor(0.0, device=self.device)

        begin_idx, end_idx = 0, 1
        for layer_idx in range(self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty += self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2) * _path_prob
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        return _mu.view(batch_size, self.leaf_node_num_), _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        batch_size = _mu.size(0)
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))
        penalty = torch.tensor(0.0, device=self.device)
        coeff = self.penalty_list[layer_idx]

        for node in range(2 ** (layer_idx + 1)):
            denom = torch.sum(_mu[:, node // 2])
            alpha = 0.5 if denom.item() == 0 else torch.sum(_path_prob[:, node] * _mu[:, node // 2]) / denom
            alpha = torch.clamp(alpha, 1e-6, 1.0 - 1e-6)
            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))
        return penalty

    def _data_augment(self, X):
        bias = torch.ones(X.size(0), 1, device=self.device)
        return torch.cat([bias, X.to(self.device)], dim=1)

    def _validate_parameters(self):
        if self.depth <= 0:
            raise ValueError(f"Depth must be positive, got {self.depth}")
        if self.lamda < 0:
            raise ValueError(f"Lambda must be non-negative, got {self.lamda}")


# In[ ]:


class RandomForestSDT:
    def __init__(self, input_dim, output_dim, depth, lamda, n_trees, subset_size, use_cuda=False, lr=1e-3, weight_decay=5e-4, epochs=50):
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.subset_size = subset_size
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = output_dim
        self.n_estimators = n_trees

        # Initialize trees (SDT models)
        self.estimators_ = nn.ModuleList([
            SDT(input_dim, output_dim, depth, lamda, use_cuda).to(self.device)
            for _ in range(n_trees)
        ])

        # Initialize learnable weights for aggregation layer
        self.weights = torch.nn.Parameter(torch.ones(n_trees, device=self.device))
        self.bias = torch.nn.Parameter(torch.zeros(1, self.num_classes, device=self.device))

    def forward(self, X):
        outputs = torch.stack([tree(X) for tree in self.estimators_], dim=0)
        return torch.mean(outputs, dim=0)

    def bootstrap_sample(self, dataset):
        indices = random.choices(range(len(dataset)), k=len(dataset))
        return Subset(dataset, indices)

    def train(self, train_loader):
        for i, tree in enumerate(self.estimators_):
            print(f"Training tree {i + 1}/{self.n_trees}")
            optimizer = torch.optim.Adam(tree.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            tree.train()

            bootstrap_data = self.bootstrap_sample(train_loader.dataset)
            bootstrap_loader = DataLoader(bootstrap_data, batch_size=train_loader.batch_size, shuffle=True)

            for epoch in tqdm(range(self.epochs), desc=f"Epochs for tree {i + 1}"):
                correct, total, running_loss = 0, 0, 0
                for data, target in bootstrap_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    feature_subset = sample(range(self.input_dim), self.subset_size)
                    output, penalty = tree(data, is_training_data=True)
                    loss = self.criterion(output, target) + penalty

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Accuracy on training data
                    _, predicted = torch.max(output, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    running_loss += loss.item()

                train_accuracy = (correct / total) * 100
                avg_loss = running_loss / len(bootstrap_loader)

            print(f"Tree {i + 1} - Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    def predict(self, data_loader):
        all_preds = []
        for data, _ in data_loader:
            data = data.to(self.device)
            with torch.no_grad():
                predictions = torch.stack([F.softmax(tree(data), dim=1) for tree in self.estimators_], dim=0)
                final_prediction = F.softmax(torch.mean(predictions, dim=0), dim=1)
                pred = final_prediction.argmax(1)
                all_preds.extend(pred.cpu().numpy())
        return np.array(all_preds)

    def evaluate(self, data_loader, true_labels):
        preds = self.predict(data_loader)
        accuracy = (preds == true_labels).mean() * 100
        precision = precision_score(true_labels, preds, average='weighted')
        recall = recall_score(true_labels, preds, average='weighted')
        f1 = f1_score(true_labels, preds, average='weighted')
        conf_matrix = confusion_matrix(true_labels, preds)

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        return accuracy, precision, recall, f1, conf_matrix

    def fit(self, X, y):
        X = X.clone().detach().to(dtype=torch.float32, device=self.device) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32, device=self.device)
        y = y.clone().detach().to(dtype=torch.long, device=self.device) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long, device=self.device)
        train_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
        self.train(train_loader)
        return self  

    def apply(self, X):
        X = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        X = X.astype(np.float32) 
        leaf_ids = [torch.argmax(tree._forward(torch.tensor(X, dtype=torch.float32, device=tree.device))[0], dim=1) for tree in self.estimators_]
        return np.array(leaf_ids).T

    def compute_tree_errors(self, X, y):
        """
        Compute the error of each tree individually
        """
        tree_errors = []
        data_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False)

        for i, tree in enumerate(self.estimators_):
            preds = []
            for data, _ in data_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    output = tree(data)
                    pred = output.argmax(dim=1)  # Get class with highest probability
                    preds.extend(pred.cpu().numpy())

            preds = np.array(preds)
            accuracy = (preds == y.cpu().numpy()).mean() * 100
            tree_errors.append(100 - accuracy)  # Error = 100 - accuracy

            print(f"Tree {i + 1}: Accuracy = {accuracy:.2f}%, Error = {100 - accuracy:.2f}%")

        return tree_errors

    def get_tree_reliability(self, X, y):
        tree_errors = self.compute_tree_errors(X, y)
        tree_errors = np.array(tree_errors, dtype=np.float32)
        eps = 1e-6
        tree_reliability = 1.0 / (tree_errors / 100.0 + eps) # Higher reliability for trees with lower error
        return tree_reliability


# In[ ]:


class TaskType(Enum):
    CLASSIFICATION = auto()


class ForestType(NamedTuple):
    kind: RandomForestSDT
    task: TaskType


FORESTS = {
    ForestType(RandomForestSDT, TaskType.CLASSIFICATION): "RandomForestSDT", 
}


class ClfRegHot:
    def fit(self, X, y) -> 'ClfRegHot':
        pass

    def refit(self, X, y) -> 'ClfRegHot':
        pass

    def optimize_weights(self, X, y) -> 'ClfRegHot':
        pass

    def predict(self, X) -> np.ndarray:
        pass

    def predict_original(self, X) -> np.ndarray:
        pass


class AFParams(NamedTuple):
    kind: RandomForestSDT
    task: TaskType
    loss_ord: int = 2
    eps: Optional[int] = None
    discount: Optional[float] = None
    forest: dict = {}


class LeafData(NamedTuple):
    xs: np.ndarray
    y: np.ndarray


def _convert_labels_to_probas(y, encoder=None):
    if y.ndim == 2 and y.shape[1] >= 2:
        return y, encoder
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape((-1, 1)))
    else:
        y = encoder.transform(y.reshape((-1, 1)))
    return y, encoder


class AttentionForest(ClfRegHot):
    def __init__(self, params: AFParams):
        self.params = params
        self.forest = None
        self._after_init()

    def _after_init(self):
        self.onehot_encoder = None

    def _make_leaf_data(self, leaf_id, tree_leaf_ids, xs, y):
        mask = (tree_leaf_ids == leaf_id)
        return LeafData(xs=xs[mask], y=y[mask])

    def _preprocess_target(self, y):
        if self.params.task == TaskType.CLASSIFICATION:
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
        return y

    def fit(self, X, y) -> 'AttentionForest':
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        
        # Provide values for depth, lamda, and subset_size
        self.forest = RandomForestSDT(input_dim=X.shape[1], output_dim=len(np.unique(y)), n_trees=100, depth=3, lamda=1e-3, subset_size=3)
        self.forest.fit(X, y)
        self.training_xs, self.training_y = X, self._preprocess_target(y)
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        self.leaf_data_x, self.leaf_data_y = self._prepare_leaf_data_fast(self.training_xs, self.training_y, self.training_leaf_ids)
        self.tree_weights = np.ones(self.forest.n_estimators)
        self.static_weights = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        return self

    def _prepare_leaf_data_fast(self, xs, y, leaf_ids):
        """
        Precompute mean feature vectors and target values per leaf for each tree.
        If a leaf has no samples, use global mean to avoid NaNs.
        """
        max_leaf_id = leaf_ids.max().item()
        y_len = 1 if y.ndim == 1 else y.shape[1]
    
        # Initialize arrays
        result_x = np.zeros((self.forest.n_estimators, max_leaf_id + 1, xs.shape[1]), dtype=np.float32)
        result_y = np.zeros((self.forest.n_estimators, max_leaf_id + 1, y_len), dtype=np.float32)
    
        # Global means in case a leaf has no samples
        global_mean_x = xs.mean(axis=0)
        global_mean_y = y.mean(axis=0) if y.ndim > 1 else np.array([y.mean()])
    
        for tree_id in range(self.forest.n_estimators):
            for leaf_id in range(max_leaf_id + 1):
                mask = (leaf_ids[:, tree_id] == leaf_id)
                if mask.any():
                    result_x[tree_id, leaf_id] = xs[mask].mean(axis=0)
                    result_y[tree_id, leaf_id] = y[mask].mean(axis=0)
                else:
                    result_x[tree_id, leaf_id] = global_mean_x
                    result_y[tree_id, leaf_id] = global_mean_y
    
        return result_x, result_y


    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
            if torch.is_tensor(X):
                X_np = X.detach().cpu().numpy()
            else:
                X_np = np.asarray(X)
            leaf_ids = self.forest.apply(X)
            all_dynamic_weights = []
            all_y = []
            for cur_x, cur_leaf_ids in zip(X, leaf_ids):
                tree_dynamic_weights, tree_dynamic_y = [], []
                for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                    leaf_mean_x, leaf_mean_y = self.leaf_data_x[cur_tree_id][cur_leaf_id], self.leaf_data_y[cur_tree_id][cur_leaf_id]
                    diff = (cur_x.numpy() - leaf_mean_x.astype(np.float64))
                    tree_dynamic_weights.append(-0.5 * np.linalg.norm(diff, 2) ** 2.0)
                    tree_dynamic_y.append(leaf_mean_y)
                all_dynamic_weights.append(np.array(tree_dynamic_weights))
                all_y.append(np.array(tree_dynamic_y))
            return np.array(all_dynamic_weights), np.array(all_y)


    def predict(self, X) -> np.ndarray:
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights if self.params.eps is not None else weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)
        return predictions


def fit_forest_split(X, y, params: AFParams, pre_size: float = 0.75, seed: Optional[int] = None):
    X_pre, X_post, y_pre, y_post = train_test_split(X, y, train_size=pre_size, random_state=seed)
    model = AttentionForest(params)
    model.fit(X_pre, y_pre)
    model.optimize_weights(X_post, y_post)
    return model


# In[ ]:


class MultiHeadAttentionABRF(nn.Module):

    def __init__(self, n_trees, n_heads, input_dim, hidden_dim, device,AttentionForest, params, forest=None):
        super().__init__()
        self.n_trees = n_trees
        self.n_heads = n_heads
        self.device = device


        # Initialize and fit forest 
        if forest is not None:
            self.AttentionForest = forest
        else:
            self.AttentionForest = AttentionForest(params)


        # Learnable parameters
        self.lambda_1 = nn.Parameter(torch.ones(n_heads, device=device))
        self.W_h = nn.Linear(n_heads, 1, bias=False)  # (n_trees, n_heads)

        # Final classifier (MLP head)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 classes → binary classification
        )

    def forward(self, X, return_prob=False):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)
    
        batch_size = X.shape[0]
    
        
        all_dynamic_weights, all_y = self.AttentionForest._get_dynamic_weights_y(X)
        all_dynamic_weights = torch.tensor(all_dynamic_weights, device=self.device, dtype=torch.float32)
        all_y = torch.tensor(all_y, device=self.device, dtype=torch.float32)
    
        
        if self.tree_reliability is None:
            tree_reliability_tensor = torch.ones(self.n_trees, device=self.device, dtype=torch.float32)
        else:
            tree_reliability_tensor = torch.tensor(self.tree_reliability, device=self.device, dtype=torch.float32)
    
        
        gamma_k = self.lambda_1[:, None] * tree_reliability_tensor
        gamma_k = gamma_k.unsqueeze(0).expand(batch_size, -1, -1)
    
        
        all_dynamic_weights = all_dynamic_weights.unsqueeze(1)
        attention_heads = []
        for i in range(self.n_heads):
            # αₖʰ = softmax(δₖʰ − ||x − Aₖ(x)||² / 2τ), here τ = 1
            scores = gamma_k[:, i, :] + all_dynamic_weights.squeeze(1)  
            attention_weights = F.softmax(scores, dim=-1)
            attention_heads.append(attention_weights)
    
        attention_heads = torch.stack(attention_heads, dim=-1)
        combined_attention = self.W_h(attention_heads).squeeze(-1)
        y_final = torch.sum(combined_attention.unsqueeze(-1) * all_y, dim=1)
        logits = self.mlp(y_final)
    
        if return_prob:
            probs = torch.softmax(logits, dim=1)
            return probs, combined_attention, gamma_k
        else:
            return logits, combined_attention, gamma_k


    def predict_proba(self, X):
        """
        Return class probabilities for input X.
        Shape: (n_samples, n_classes)
        """
        self.eval()
    
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)
    
        with torch.no_grad():
            probs, _, _ = self.forward(X, return_prob=True)  
    
        return probs.detach().cpu().numpy()


    def predict(self, X):
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(X, return_prob=False)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()


# In[ ]:


class Patient:
    def __init__(self, patient_id, lead_time, no_show_prob=None, features=None):
        self.id = patient_id
        self.patient_id = patient_id 
        self.lead_time = lead_time
        self.no_show_prob = no_show_prob
        self.assigned_slot = None
        self.assigned_physician = None
        self.status = None
        self.features = features # Store the features

class Clinic:
    def __init__(self, clinic_name, departments_info):
        self.name = clinic_name
        self.departments = [Department(d["department"], d["num_physicians"]) for d in departments_info]

class Department:
    def __init__(self, dept_name, num_physician):
        self.name = dept_name
        self.physician = [Physician(i) for i in range (num_physician)]

    def get_least_loaded_physician(self):
        return min(self.physician, key=lambda p: p.total_patients())

class Physician:
    def __init__(self, physician_id, num_slots = 12):
        self.id = physician_id
        self.slots = [TimeSlot(slot_time=i) for i in range (num_slots)]

    def get_available_slot(self):
        return [slot for slot in self.slots if len(slot.scheduled_patients) < 2]

    def total_patients(self):
        return sum(len(slot.scheduled_patients) for slot in self.slots)

class TimeSlot:
    def __init__(self, slot_time):
        self.slot_time = slot_time
        self.scheduled_patients = []

    def add_patient(self, patient):
        if len(self.scheduled_patients) < 2:
            self.scheduled_patients.append(patient)
            patient.assigned_slot = self
            return True
        return False

# In[ ]:


from collections import defaultdict, deque

def build_clinics():
    clinic_structure = [
        {"clinic": "DDI2", "department": "COLORECTAL SURG CAD", "num_physicians": 3},
        {"clinic": "SSI", "department": "DENTISTRY CAD", "num_physicians": 3},
        {"clinic": "DDI1", "department": "GASTROENTEROLOGY CAD", "num_physicians": 10},
        {"clinic": "DDI2", "department": "GENERAL SURG CAD", "num_physicians": 7},
        {"clinic": "SSI", "department": "GYNECOLOGY CAD", "num_physicians": 1},
        {"clinic": "SSI", "department": "OTOLARYNGOLOGY CAD", "num_physicians": 7},
        {"clinic": "SSI", "department": "PLASTIC SURG CAD", "num_physicians": 2},
        {"clinic": "SSI", "department": "UROLOGY CAD", "num_physicians": 5},
        {"clinic": "DDI2", "department": "NUTRITION HAD", "num_physicians": 7},
        {"clinic": "DDI2", "department": "WOUND CARE HAD", "num_physicians": 1},
        {"clinic": "SSI", "department": "SPEECH THERAPY HAD", "num_physicians": 2},
    ]

    clinic_dict = defaultdict(list)
    for d in clinic_structure:
        clinic_dict[d["clinic"]].append({
            "department": d["department"],
            "num_physicians": d["num_physicians"]
        })

    clinics = []
    for clinic_name, dept_info in clinic_dict.items():
        clinics.append(Clinic(clinic_name, dept_info))

    return clinics

# In[ ]:


import torch
import pandas as pd

def mhasrf_predict_no_show_prob_batch(features_batch, scaler, model, no_show_class_index=1):
    """
    Predict no-show probabilities for a batch of patients using MHASRF.

    Args:
        features_batch: array-like, shape (batch_size, num_features)
        scaler: fitted StandardScaler
        no_show_class_index: index of the 'no-show' class (0 or 1)

    Returns:
        np.ndarray: array of no-show probabilities, shape (batch_size,)
    """
    # Convert to DataFrame with correct columns
    features_df = pd.DataFrame(features_batch, columns=scaler.feature_names_in_)
    
    # Standardize
    features_scaled = scaler.transform(features_df)
    
    # Convert to tensor
    device = next(model.parameters()).device
    x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    
    # Batch predictions
    with torch.no_grad():
        output = model(x)
    
    # Extract tensor if model returns tuple/list
    if isinstance(output, (tuple, list)):
        pred = None
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                pred = item
                break
        if pred is None:
            raise ValueError("No valid tensor prediction found in model output.")
    elif isinstance(output, torch.Tensor):
        pred = output
    else:
        raise TypeError(f"Unexpected model output type: {type(output)}")

    # Convert to probabilities
    if pred.shape[-1] == 1:
        probs = torch.sigmoid(pred).squeeze(-1).cpu().numpy()
        if no_show_class_index == 0:
            probs = 1.0 - probs
    else:
        probs = torch.softmax(pred, dim=-1)[:, no_show_class_index].cpu().numpy()
    
    return probs

def get_patient_no_show_prob(patient_row, scaler):
    """
    patient_row: pd.Series containing the features for this patient
    scaler: fitted StandardScaler used during training
    """
    # Make it a batch of 1
    features_batch = [patient_row[Features].values.tolist()]
    
    # Call the MHASRF prediction function
    prob = mhasrf_predict_no_show_prob_batch(features_batch, scaler)[0]
    
    return prob


# In[ ]:


import numpy as np
import random

# Assumed to exist in your project:
# - TimeSlot (has .scheduled_patients list and .add_patient(patient))
# - Patient
# - mhasrf_predict_no_show_prob_batch(features_batch, scaler, model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SchedulingEnv(gym.Env):
    """
    Actions (Discrete(3)):
      0 = single book
      1 = double book
      2 = reject   (ONLY allowed when no feasible slot exists)

    Observation (10 dims):
      [clinic, department, physician, appointment_date, time_slot,
       slot_status, no_show_prob, double_book, scheduled_patients_count, available_slots_count]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, clinics, df_shuffled, Features, scaler, model,
                 booking_horizon=14, lambda_booking=500, alpha=1, beta=1, gamma=1,
                 seed=None, warmup_days=14):
        super().__init__()
        self.clinics = clinics
        self.booking_horizon = int(booking_horizon)
        self.lambda_booking = float(lambda_booking)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.warmup_days = int(warmup_days)

        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self._arrivals_locked = False
        self._predef_arrivals = {}
        self._predef_start_day = 0
        self._predef_total = 0
        self._arrival_inserted_patients = set()

        self.arrivals = None
        self.deterministic = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.calendar = {}
        self.current_event_idx = 0
        self.events = []
        self.patient_seq = 0

        self._counted_patients_checked_in = set()
        self._counted_patients_showed = set()

        self.total_slots_checked_in = 0
        self.slots_with_patient_checked_in = 0
        self.double_book_total = 0
        self.double_show_count = 0
        self.double_balance_sum = 0.0

        self.df_shuffled = df_shuffled
        self.Features = Features
        self.scaler = scaler
        self.model = model

        self._build_index_maps()
        self._last_obs = None
        self._last_context = None

    def set_predefined_arrivals(self, arrivals, start_day=0):
        arr = np.asarray(arrivals, dtype=int).copy()
        if arr.ndim != 1:
            raise ValueError("arrivals must be 1D array-like.")
        if np.any(arr < 0):
            raise ValueError("arrivals must be non-negative.")
        if len(arr) < (self.booking_horizon - start_day):
            pad_len = (self.booking_horizon - start_day) - len(arr)
            arr = np.concatenate([arr, np.zeros(pad_len, dtype=int)], axis=0)

        self._predef_arrivals.clear()
        self._predef_start_day = int(start_day)
        for i, cnt in enumerate(arr[: self.booking_horizon - start_day]):
            day = start_day + i
            self._predef_arrivals[day] = int(cnt)

        self._predef_total = int(arr[: self.booking_horizon - start_day].sum())
        self._arrivals_locked = True
        return self._predef_total

    def _build_index_maps(self):
        self.clinic_index = {c.name: i for i, c in enumerate(self.clinics)}
        self.dept_index = {}
        self.phy_counts = {}
        for c in self.clinics:
            self.dept_index[c.name] = {d.name: j for j, d in enumerate(c.departments)}
            for d in c.departments:
                self.phy_counts[(c.name, d.name)] = len(d.physician)

    def _init_calendar(self):
        self.calendar.clear()
        min_day = -self.warmup_days
        max_day = self.booking_horizon - 1
        for ci, c in enumerate(self.clinics):
            for di, d in enumerate(c.departments):
                for pi, p in enumerate(d.physician):
                    num_slots = len(p.slots)
                    for day in range(min_day, max_day + 1):
                        self.calendar[(ci, di, pi, day)] = [TimeSlot(slot_time=k) for k in range(num_slots)]

    def _all_physician_keys(self):
        for ci, c in enumerate(self.clinics):
            for di, d in enumerate(c.departments):
                for pi, _ in enumerate(d.physician):
                    yield (ci, di, pi)

    def _get_least_loaded_physician_key(self, dept):
        c_name = None
        for cname, cidx in self.clinic_index.items():
            for d in self.clinics[cidx].departments:
                if d is dept:
                    c_name = cname
                    break
            if c_name is not None:
                break

        di = self.dept_index[c_name][dept.name]
        ci = self.clinic_index[c_name]

        best_key, best_load = None, float("inf")
        for pi, _p in enumerate(self.clinics[ci].departments[di].physician):
            total = sum(
                len(s.scheduled_patients)
                for day in range(self.booking_horizon)
                for s in self.calendar[(ci, di, pi, day)]
            )
            if total < best_load:
                best_load = total
                best_key = (ci, di, pi)
        return best_key

    def _count_physician_load(self, key):
        ci, di, pi = key
        return sum(
            len(s.scheduled_patients)
            for day in range(self.booking_horizon)
            for s in self.calendar[(ci, di, pi, day)]
        )

    def _available_slots_for_key_day(self, key, day):
        ci, di, pi = key
        slots = self.calendar[(ci, di, pi, day)]
        return [k for k, s in enumerate(slots) if len(s.scheduled_patients) < 2]

    def _schedule_patient(self, patient, key, day, slot_idx):
        ci, di, pi = key
        slot = self.calendar[(ci, di, pi, day)][slot_idx]
        if len(slot.scheduled_patients) < 2:
            slot.add_patient(patient)
            return True
        return False

    def _sample_lead_time(self):
        lt = self.rng.gamma(shape=0.91, scale=8.11)
        return int(np.clip(round(lt), 1, self.booking_horizon))

    def _compute_reward(self, event):
        Ut_good = 0.0
        Dt_good = 1.0
        Bt_good = 0.0

        if event["type"] == "arrival":
            patient = event["patient"]
            ci, di, pi = event["physician_key"]
            day = event["day"]
            slot_idx = event["slot_idx"]
            slot = self.calendar[(ci, di, pi, day)][slot_idx]

            actual_shows = sum(1 for p in slot.scheduled_patients if p.status == "showed")
            Ut_good = max(0.0, 1.0 - abs(actual_shows - 1.0))

            if len(slot.scheduled_patients) == 2:
                s1 = slot.scheduled_patients[0].status
                s2 = slot.scheduled_patients[1].status
                raw_Dt = 1.0 if (s1 == "showed" and s2 == "showed") else 0.0
                Dt_good = 1.0 - raw_Dt

                p1 = slot.scheduled_patients[0].no_show_prob
                p2 = slot.scheduled_patients[1].no_show_prob
                expected_attendance = (1.0 - p1) + (1.0 - p2)
                Bt_good = max(0.0, 1.0 - abs(expected_attendance - 1.0))

            elif len(slot.scheduled_patients) == 1:
                p_single = slot.scheduled_patients[0].no_show_prob
                expected_attendance = (1.0 - p_single)
                Bt_good = max(0.0, 1.0 - abs(expected_attendance - 1.0))
                Dt_good = 1.0

            else:
                Ut_good = 0.0
                Dt_good = 1.0
                Bt_good = 0.0

        return np.array([Ut_good, Dt_good, Bt_good], dtype=np.float32)

    def _compute_booking_shaped(self, phy_key, appt_day, slot_idx, patient):
        slot = self.calendar[(*phy_key, appt_day)][slot_idx]
        existing_probs = [p.no_show_prob for p in slot.scheduled_patients] if slot.scheduled_patients else []
        avg_existing_p = float(np.mean(existing_probs)) if existing_probs else 0.5

        p_main = float(patient.no_show_prob)

        Ut_single_good = (1.0 - p_main)
        Ut_double_good = (1.0 - p_main) * avg_existing_p + p_main * (1.0 - avg_existing_p)

        Dt_single_good = 1.0
        exp_double_show_prob = (1.0 - p_main) * (1.0 - avg_existing_p)
        Dt_double_good = 1.0 - float(np.clip(exp_double_show_prob, 0.0, 1.0))

        Bt_single_good = max(0.0, 1.0 - abs((1.0 - p_main) - 1.0))
        exp_att_double = (1.0 - p_main) + (1.0 - avg_existing_p)
        Bt_double_good = max(0.0, 1.0 - abs(exp_att_double - 1.0))

        # IMPORTANT CHANGE:
        # reject should NOT look like single-book; make it neutral/penalty.
        reject = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # or [-0.05, 0.0, 0.0]

        return {
            0: np.array([Ut_single_good, Dt_single_good, Bt_single_good], dtype=np.float32),
            1: np.array([Ut_double_good, Dt_double_good, Bt_double_good], dtype=np.float32),
            2: reject
        }

    def _generate_booking_events(self, start_day=0, num_days=None, prob_noise_std=0.05,
                                 prob_perturb=None, perturb_mode="absolute", perturb_sign=None):
        if num_days is None:
            num_days = self.booking_horizon

        seq = 0
        created_per_day = {}
        warmup_mode = start_day < 0

        valid_rows = self.df_shuffled.dropna(subset=self.Features)
        if len(valid_rows) == 0:
            raise ValueError("No valid rows available in df_shuffled after dropping missing Features!")

        for day in range(start_day, start_day + num_days):

            if warmup_mode:
                n = max(int(self.lambda_booking * 0.1), self.rng.poisson(self.lambda_booking))
            else:
                if self._arrivals_locked:
                    n = int(self._predef_arrivals.get(day, 0))
                else:
                    n = self.rng.poisson(self.lambda_booking)

            n = min(int(n), len(valid_rows))
            if n <= 0:
                created_per_day[day] = 0
                continue

            batch_rows = valid_rows.sample(n=n, replace=False, random_state=None)
            features_batch = batch_rows[self.Features].values

            predicted_probs = mhasrf_predict_no_show_prob_batch(features_batch, self.scaler, self.model)
            predicted_probs = np.nan_to_num(predicted_probs, nan=0.5, posinf=1.0, neginf=0.0)

            day_patients = []
            for i in range(n):
                patient_id = self.patient_seq
                self.patient_seq += 1

                lead_time = self._sample_lead_time()

                patient = Patient(
                    patient_id,
                    lead_time,
                    no_show_prob=float(predicted_probs[i]),
                    features=batch_rows.iloc[i][self.Features].to_dict(),
                )
                patient.patient_id = patient_id
                patient.created_from_warmup = warmup_mode

                p = float(predicted_probs[i])
                if prob_perturb is not None:
                    sign = 1 if perturb_sign is None else int(perturb_sign)
                    if perturb_mode == "absolute":
                        p = p + sign * float(prob_perturb)
                    elif perturb_mode == "relative":
                        p = p * (1.0 + sign * float(prob_perturb))
                    else:
                        raise ValueError("perturb_mode must be 'absolute' or 'relative'.")
                patient.no_show_prob = float(np.clip(np.nan_to_num(p, nan=0.5), 0.01, 1.0))

                all_departments = [d for c in self.clinics for d in c.departments]
                patient.department = self.rng.choice(all_departments)

                day_patients.append(patient)

            created_per_day[day] = n

            for patient in day_patients:
                self.events.append(
                    {
                        "type": "booking",
                        "day": day,
                        "patient": patient,
                        "department": patient.department,
                        "seq": seq,
                    }
                )
                seq += 1

        self.events.sort(key=lambda e: (e["day"], e.get("seq", 0)))

        if self._arrivals_locked and not warmup_mode:
            locked_created = sum(created_per_day.get(d, 0) for d in range(self._predef_start_day, self.booking_horizon))
            assert locked_created == self._predef_total, (
                f"[SchedulingEnv] Mismatch: created {locked_created} vs locked {self._predef_total}\n"
                f"Per-day breakdown: {[created_per_day.get(d,0) for d in range(self._predef_start_day, self.booking_horizon)]}\n"
                f"Expected: {[int(self._predef_arrivals.get(d, 0)) for d in range(self._predef_start_day, self.booking_horizon)]}"
            )

    def _insert_arrival_event(self, patient, physician_key, day, slot_idx, from_warmup=False):
        pid = getattr(patient, "patient_id", None)
        if pid in self._arrival_inserted_patients:
            return

        self._arrival_inserted_patients.add(pid)

        e = {
            "type": "arrival",
            "day": day,
            "patient": patient,
            "physician_key": physician_key,
            "slot_idx": slot_idx,
            "seq": slot_idx,
            "from_warmup": bool(from_warmup),
        }
        self.events.append(e)
        self.events.sort(key=lambda ev: (ev["day"], 0 if ev["type"] == "booking" else 1, ev["seq"]))

    def _run_warmup(self):
        if self.warmup_days <= 0:
            return

        self._generate_booking_events(start_day=-self.warmup_days, num_days=self.warmup_days)

        for ev in list(self.events):
            if ev["type"] != "booking" or ev["day"] >= 0:
                continue

            patient = ev["patient"]
            dept = ev["department"]

            if not hasattr(patient, "created_from_warmup"):
                patient.created_from_warmup = True

            phy_key = self._get_least_loaded_physician_key(dept)
            appt_day = int(np.clip(ev["day"] + patient.lead_time, -self.warmup_days, -1))

            avail = self._available_slots_for_key_day(phy_key, appt_day)
            if not avail:
                continue

            empty = [k for k in avail if len(self.calendar[(*phy_key, appt_day)][k].scheduled_patients) == 0]
            slot_idx = random.choice(empty) if empty else random.choice(avail)

            ok = self._schedule_patient(patient, phy_key, appt_day, slot_idx)
            if ok:
                patient.assigned_physician = phy_key
                patient.assigned_slot = slot_idx
                patient.appointment_day = appt_day

        # remove warmup patients from episode days >= 0
        for ci, di, pi in self._all_physician_keys():
            for day in range(0, self.booking_horizon):
                slots = self.calendar[(ci, di, pi, day)]
                for slot in slots:
                    if getattr(slot, "scheduled_patients", None):
                        slot.scheduled_patients = [
                            p for p in slot.scheduled_patients
                            if not getattr(p, "created_from_warmup", False)
                        ]

        # remove any arrival events in episode window
        self.events = [e for e in self.events if not (e["type"] == "arrival" and e.get("day", 0) >= 0)]
        self.events.sort(key=lambda e: (e["day"], 0 if e["type"] == "booking" else 1, e.get("seq", 0)))
        self.current_event_idx = 0

    def reset(self, *, seed=None, options=None, arrivals=None, deterministic=False,
              prob_perturb=None, perturb_mode="absolute", perturb_sign=None):
        super().reset(seed=seed)

        if deterministic:
            self.rng = np.random.default_rng(42)
            random.seed(42)
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)

        if arrivals is not None:
            self.arrivals = np.asarray(arrivals, dtype=int).copy()
            self.set_predefined_arrivals(self.arrivals, start_day=0)
        else:
            self.arrivals = self.rng.poisson(self.lambda_booking, size=self.booking_horizon)
            self.set_predefined_arrivals(self.arrivals, start_day=0)

        self._init_calendar()
        self.patient_seq = 0
        self.events.clear()
        self.current_event_idx = 0

        self.total_slots_checked_in = 0
        self.slots_with_patient_checked_in = 0
        self.double_book_total = 0
        self.double_show_count = 0
        self.double_balance_sum = 0.0
        self._arrival_inserted_patients.clear()

        prev_lock_state = self._arrivals_locked
        self._arrivals_locked = False
        self._run_warmup()
        self._arrivals_locked = prev_lock_state

        self.prob_perturb = prob_perturb
        self.perturb_mode = perturb_mode
        self.perturb_sign = perturb_sign

        self._generate_booking_events(
            start_day=0,
            num_days=self.booking_horizon,
            prob_perturb=self.prob_perturb,
            perturb_mode=self.perturb_mode,
            perturb_sign=self.perturb_sign
        )

        self._counted_patients_checked_in.clear()
        self._counted_patients_showed.clear()

        self._debug_sanity_after_reset()

        return np.zeros(10, dtype=np.float32), {}

    # --------------------------
    # ACTION MASK (CRUCIAL)
    # --------------------------
    def get_action_mask(self):
        """
        1 = allowed, 0 = disallowed.
        If booking feasible -> [1, 1, 0]  (reject NOT allowed)
        If not feasible     -> [0, 0, 1]  (only reject allowed)
        """
        if self.current_event_idx >= len(self.events):
            return np.array([1, 1, 1], dtype=np.int8)

        ev = self.events[self.current_event_idx]
        if ev["type"] != "booking":
            return np.array([1, 1, 1], dtype=np.int8)

        patient = ev["patient"]
        dept = ev["department"]
        phy_key = self._get_least_loaded_physician_key(dept)
        appt_day = int(np.clip(ev["day"] + patient.lead_time, 0, self.booking_horizon - 1))

        def find_spillover_slot(start_day):
            for future_day in range(start_day, self.booking_horizon):
                avail = self._available_slots_for_key_day(phy_key, future_day)
                if avail:
                    return future_day, avail
            return None, []

        future_day, avail = find_spillover_slot(appt_day)
        if not avail:
            future_day, avail = find_spillover_slot(0)

        if not avail:
            alt_phy_key, alt_day, alt_avail = self._find_alternative_physician_slot(dept, start_day=appt_day)
            if alt_avail:
                avail = alt_avail

        if avail:
            return np.array([1, 1, 0], dtype=np.int8)
        return np.array([0, 0, 1], dtype=np.int8)

    def step(self, action):
        # Terminal check
        if self.current_event_idx >= len(self.events):
            zero_obs = np.zeros(10, dtype=np.float32)
            zero_rew = np.zeros(3, dtype=np.float32)
            return zero_obs, zero_rew, True, False, {"event_type": "terminal"}

        event = self.events[self.current_event_idx]

        # Booking Step
        if event["type"] == "booking":
            patient = event["patient"]
            dept = event["department"]
            phy_key = self._get_least_loaded_physician_key(dept)

            appt_day = int(np.clip(event["day"] + patient.lead_time, 0, self.booking_horizon - 1))
            reward_vector = np.zeros(3, dtype=np.float32)
            scheduled = False
            chosen_slot_idx = None

            def find_spillover_slot(start_day):
                for future_day in range(start_day, self.booking_horizon):
                    avail = self._available_slots_for_key_day(phy_key, future_day)
                    if avail:
                        return future_day, avail
                return None, []

            future_day, avail = find_spillover_slot(appt_day)
            if not avail:
                future_day, avail = find_spillover_slot(0)

            if not avail:
                alt_phy_key, alt_day, alt_avail = self._find_alternative_physician_slot(dept, start_day=appt_day)
                if alt_avail:
                    phy_key = alt_phy_key
                    future_day = alt_day
                    avail = alt_avail
                else:
                    # No feasible slot anywhere in dept -> forced reject-like outcome
                    self.current_event_idx += 1
                    obs = self._build_observation_for_next_event()
                    terminated = self.current_event_idx >= len(self.events)
                    return obs, reward_vector, terminated, False, {
                        "event_type": "booking",
                        "action": int(action),
                        "scheduled": False,
                        "reason": "no_available_slot_in_department",
                        "patient_id": patient.patient_id,
                        "patient_obj": patient
                    }

            # IMPORTANT:
            # If your PPO uses action masking, it will never pass action==2 when avail exists.
            # Still, keep reject behavior explicit.
            if action == 2:
                # reject: do not schedule
                self.current_event_idx += 1
                obs = self._build_observation_for_next_event()
                terminated = self.current_event_idx >= len(self.events)
                return obs, np.zeros(3, dtype=np.float32), terminated, False, {
                    "event_type": "booking",
                    "action": int(action),
                    "scheduled": False,
                    "reason": "reject",
                    "patient_id": patient.patient_id,
                    "patient_obj": patient
                }

            # choose slot for action 0/1
            if action == 0:  # single book
                empty = [k for k in avail if len(self.calendar[(*phy_key, future_day)][k].scheduled_patients) == 0]
                chosen_slot_idx = random.choice(empty) if empty else random.choice(avail)

            elif action == 1:  # double book
                patient.double_booked = True
                singly = [k for k in avail if len(self.calendar[(*phy_key, future_day)][k].scheduled_patients) == 1]
                chosen_slot_idx = random.choice(singly) if singly else random.choice(avail)
            else:
                chosen_slot_idx = random.choice(avail)

            booking_shaped = self._compute_booking_shaped(phy_key, future_day, chosen_slot_idx, patient)
            reward_vector = booking_shaped[int(action)]

            ok = self._schedule_patient(patient, phy_key, future_day, chosen_slot_idx)

            if ok:
                patient.assigned_physician = phy_key
                patient.assigned_slot = chosen_slot_idx
                patient.appointment_day = future_day

                pid = getattr(patient, "patient_id", None)
                if pid is None:
                    raise ValueError("no patient id found")

                if pid not in self._arrival_inserted_patients:
                    self._insert_arrival_event(patient, phy_key, future_day, chosen_slot_idx)
                    patient.arrival_inserted = True

                scheduled = True

            self.current_event_idx += 1
            obs = self._build_observation_for_next_event()
            terminated = self.current_event_idx >= len(self.events)
            info = {
                "event_type": "booking",
                "action": int(action),
                "scheduled": bool(scheduled),
                "physician_key": phy_key,
                "appt_day": int(future_day) if future_day is not None else None,
                "slot_idx": int(chosen_slot_idx) if chosen_slot_idx is not None else None,
                "shaped_reward": reward_vector.copy(),
                "realized_reward": None,
                "patient_id": patient.patient_id,
                "patient_obj": patient
            }
            return obs, reward_vector, terminated, False, info

        # Arrival
        elif event["type"] == "arrival":
            patient = event["patient"]
            ci, di, pi = event["physician_key"]
            day = event["day"]
            slot_idx = event["slot_idx"]
            slot = self.calendar[(ci, di, pi, day)][slot_idx]
            from_warmup = bool(event.get("from_warmup", False))

            if patient.status is None:
                patient.status = "showed" if self.rng.random() > patient.no_show_prob else "no-show"

            if patient.patient_id not in self._counted_patients_checked_in:
                self.total_slots_checked_in += 1
                self._counted_patients_checked_in.add(patient.patient_id)

            if patient.status == "showed" and patient.patient_id not in self._counted_patients_showed:
                self.slots_with_patient_checked_in += 1
                self._counted_patients_showed.add(patient.patient_id)

            double_show = False
            if len(slot.scheduled_patients) == 2:
                self.double_book_total += 1
                p1 = slot.scheduled_patients[0].no_show_prob
                p2 = slot.scheduled_patients[1].no_show_prob
                s1 = slot.scheduled_patients[0].status
                s2 = slot.scheduled_patients[1].status
                if s1 == "showed" and s2 == "showed":
                    self.double_show_count += 1
                    double_show = True
                self.double_balance_sum += -abs((1 - p1) + (1 - p2) - 1)

            reward_vector = self._compute_reward(event)

            self.current_event_idx += 1
            obs = self._build_observation_for_next_event()
            terminated = self.current_event_idx >= len(self.events)
            info = {
                "event_type": "arrival",
                "physician_key": (ci, di, pi),
                "day": int(day),
                "slot_idx": int(slot_idx),
                "slot_occupancy": len(slot.scheduled_patients),
                "patient_status": patient.status,
                "double_show": double_show,
                "realized_reward": reward_vector.copy(),
                "shaped_reward": None,
                "patient_id": patient.patient_id,
                "patient_obj": patient,
                "from_warmup": from_warmup
            }
            return obs, reward_vector, terminated, False, info

        # Fallback
        else:
            self.current_event_idx += 1
            obs = self._build_observation_for_next_event()
            term = self.current_event_idx >= len(self.events)
            return obs, np.zeros(3, dtype=np.float32), term, False, {"event_type": "unknown"}

    def _build_observation_for_next_event(self):
        if self.current_event_idx >= len(self.events):
            self._last_obs = np.zeros(10, dtype=np.float32)
            self._last_context = None
            return self._last_obs

        ev = self.events[self.current_event_idx]

        if ev["type"] == "booking":
            patient = ev["patient"]
            dept = ev["department"]
            phy_key = self._get_least_loaded_physician_key(dept)
            day = int(np.clip(ev["day"] + patient.lead_time, 0, self.booking_horizon - 1))

            avail = self._available_slots_for_key_day(phy_key, day)
            empty_slots = [k for k in avail if len(self.calendar[(*phy_key, day)][k].scheduled_patients) == 0]

            if empty_slots:
                slot_idx = random.choice(empty_slots)
            elif avail:
                slot_idx = random.choice(avail)
            else:
                slot_idx = 0

            ci, di, pi = phy_key
            slots = self.calendar[(ci, di, pi, day)]
            slot = slots[slot_idx]

            obs = np.array([
                float(ci), float(di), float(pi), float(day), float(slot_idx),
                float(len(slot.scheduled_patients)), float(patient.no_show_prob),
                float(1 if len(slot.scheduled_patients) >= 2 else 0),
                float(self._count_physician_load(phy_key)),
                float(len([s for s in slots if len(s.scheduled_patients) < 2]))
            ], dtype=np.float32)

            self._last_obs = obs
            self._last_context = {"kind": "booking", "phy_key": phy_key, "day": day, "slot_idx": slot_idx}
            return obs

        else:
            patient = ev["patient"]
            ci, di, pi = phy_key = ev["physician_key"]
            day = ev["day"]
            slot_idx = ev["slot_idx"]

            slots = self.calendar[(ci, di, pi, day)]
            slot = slots[slot_idx]

            obs = np.array([
                float(ci), float(di), float(pi), float(day), float(slot_idx),
                float(len(slot.scheduled_patients)), float(patient.no_show_prob),
                float(1 if len(slot.scheduled_patients) >= 2 else 0),
                float(self._count_physician_load(phy_key)),
                float(len([s for s in slots if len(s.scheduled_patients) < 2]))
            ], dtype=np.float32)

            self._last_obs = obs
            self._last_context = {"kind": "arrival", "phy_key": phy_key, "day": day, "slot_idx": slot_idx}
            return obs

    def render(self, mode="human"):
        print(f"Events remaining: {len(self.events) - self.current_event_idx}")

    def _debug_sanity_after_reset(self):
        # Your existing debug checks can live here (you currently don't print/assert anything).
        return

    def _find_alternative_physician_slot(self, dept, start_day=0):
        """
        Try to find an available slot for this department across *all* physicians.
        Returns (phy_key, day, avail_slots) or (None, None, []) if none found.
        """
        c_name = None
        for cname, cidx in self.clinic_index.items():
            for d in self.clinics[cidx].departments:
                if d is dept:
                    c_name = cname
                    break
            if c_name is not None:
                break

        ci = self.clinic_index[c_name]
        di = self.dept_index[c_name][dept.name]

        for pi, _p in enumerate(self.clinics[ci].departments[di].physician):
            for future_day in range(0, self.booking_horizon):
                avail = self._available_slots_for_key_day((ci, di, pi), future_day)
                if avail:
                    return (ci, di, pi), future_day, avail

        return None, None, []


# In[ ]:


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        return logits


# --- Value network ---
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)



# In[ ]:


# --- Replay memory ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'log_prob'))

class ReplayMemory:
    def __init__(self, capacity=1000000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def __iter__(self):
        return iter(self.memory)
    
    def clear(self):
        self.memory.clear()


# In[ ]:


class MPPPOAgent:
    def __init__(self, state_dim, action_dim, weight_vector, gamma=0.99, eps_clip=0.2, lr_policy=1e-4, lr_value=1e-3):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.w_vector = torch.FloatTensor(weight_vector).to(self.device)  # weight vector for scalarization
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value = ValueNetwork(state_dim).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr_value)

        self.memory = ReplayMemory()

    def select_action(self, state, action_mask=None, deterministic=False):
        obs = state[0] if isinstance(state, tuple) else state
        state_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
    
        logits = self.policy_old(state_tensor)  # [1, A]
    
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).view(1, -1)  # [1, A]
            # set disallowed actions to very negative logits
            logits = logits.masked_fill(~mask, -1e9)
    
        dist = torch.distributions.Categorical(logits=logits)
    
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
    
        log_prob = dist.log_prob(action).detach()
        return int(action.item()), log_prob


    def store_transition(self, state, action, reward_vector, next_state, done, log_prob):
        state_array = np.asarray(state[0] if isinstance(state, tuple) else state, dtype=np.float32)
        next_state_array = np.asarray(next_state[0] if isinstance(next_state, tuple) else next_state, dtype=np.float32)
        reward_vector = np.asarray(reward_vector, dtype=np.float32)
        self.memory.push(state_array, action, reward_vector, next_state_array, done, log_prob)

    def scalarize_reward(self, reward_vector):
        reward_vector = np.array(reward_vector, dtype=np.float32)
        return float(np.dot(self.w_vector.cpu().numpy(), reward_vector))

    def update(
        self,
        K=4,
        batch_size=64,
        clip_grad_norm=0.5,
        current_epoch=1,
        total_epochs=50,
        gae_lambda=0.95  # set to None to use (returns - values) without GAE
    ):
        """
        PPO update with strict shape handling, minibatching, and (optional) GAE.
        Assumes self.memory stores tuples: (state, action, reward_vec, next_state, done, old_log_prob)
        """
    
        # --------- 1) Pull & scalarize transitions ----------
        transitions = list(self.memory)
        if len(transitions) == 0:
            # Nothing to update; avoid zero-division
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    
        states, actions, rewards_vecs, next_states, dones, old_log_probs = zip(*transitions)
    
        # scalarize rewards via weight vector
        scalar_rewards = [self.scalarize_reward(r) for r in rewards_vecs]
    
        # to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)          # [N, state_dim]
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)# [N, state_dim]
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).view(-1)  # [N]
        rewards = torch.tensor(np.array(scalar_rewards), dtype=torch.float32, device=self.device).view(-1)  # [N]
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).view(-1)   # [N]
    
        # old_log_probs may have come in as 0-dim tensors; make float vector
        old_log_probs = torch.tensor(
            np.array([float(lp) for lp in old_log_probs], dtype=np.float32),
            device=self.device
        ).view(-1)  # [N]
    
        N = states.size(0)
    
        # --------- 2) Compute values and advantages ----------
        with torch.no_grad():
            values = self.value(states).view(-1)          # [N]
            next_values = self.value(next_states).view(-1)  # [N]
    
            if gae_lambda is not None:
                # GAE(λ)
                deltas = rewards + self.gamma * next_values * (1.0 - dones) - values  # [N]
                advantages = torch.zeros_like(deltas)
                gae = 0.0
                # compute from last to first
                for t in reversed(range(N)):
                    mask = 1.0 - dones[t]
                    gae = deltas[t] + self.gamma * gae_lambda * mask * gae
                    advantages[t] = gae
                returns = advantages + values
            else:
                # no GAE: simple Monte Carlo style bootstrap-one-step
                returns = rewards + self.gamma * next_values * (1.0 - dones)  # [N]
                advantages = returns - values                                  # [N]
    
            # normalize advantages (robust to tiny std)
            adv_mean, adv_std = advantages.mean(), advantages.std()
            if torch.isnan(adv_std) or adv_std < 1e-6:
                adv_std = torch.tensor(1.0, device=advantages.device)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    
        # Safety: enforce exact shapes to avoid broadcasting surprises
        assert actions.shape == (N,), f"actions shape {actions.shape} != ({N},)"
        assert old_log_probs.shape == (N,), f"old_log_probs shape {old_log_probs.shape} != ({N},)"
        assert advantages.shape == (N,), f"advantages shape {advantages.shape} != ({N},)"
        assert returns.shape == (N,), f"returns shape {returns.shape} != ({N},)"
    
        # Precompute entropy decay
        entropy_coef = max(0.01 * (1 - current_epoch / max(1, total_epochs)), 0.001)
    
        # --------- 3) PPO epochs with minibatches ----------
        idxs = torch.arange(N, device=self.device)
    
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_mb = 0
    
        for _ in range(K):
            # Shuffle indices each epoch
            perm = idxs[torch.randperm(N, device=self.device)]
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                mb_idx = perm[start:end]
    
                mb_states = states[mb_idx].view(len(mb_idx), -1)  # [B, state_dim]
                mb_actions = actions[mb_idx]         # [B]
                mb_old_logp = old_log_probs[mb_idx]  # [B]
                mb_advs = advantages[mb_idx]         # [B]
                mb_returns = returns[mb_idx]         # [B]
    
                # Policy forward
                # Policy forward (per minibatch)
                logits = self.policy(mb_states)                  # [B, action_dim]
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_actions)                 # [B]
                logp = logp.view(-1)                             # flatten
                mb_old_logp = mb_old_logp.view(-1)               # flatten
                entropy = dist.entropy().mean()
                
                # Ratio & clipped surrogate
                assert logp.shape == mb_old_logp.shape, f"logp {logp.shape} vs mb_old_logp {mb_old_logp.shape}"
                ratio = torch.exp(logp - mb_old_logp)            # [B]
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
    
                # Value loss
                values_pred = self.value(mb_states).view(-1)     # [B]
                value_loss = (mb_returns - values_pred).pow(2).mean()
    
                # Backprop (policy)
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad_norm)
                self.optimizer_policy.step()
    
                # Backprop (value)
                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), clip_grad_norm)
                self.optimizer_value.step()
    
                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy += float(entropy.detach().cpu())
                total_mb += 1
    
        # Sync old policy with new one (standard PPO practice)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
        # Always clear memory after update to avoid accumulation
        self.memory.clear()
    
        # Averages for logging
        return {
            'policy_loss': total_policy_loss / max(1, total_mb),
            'value_loss': total_value_loss / max(1, total_mb),
            'entropy': total_entropy / max(1, total_mb)
        }

    def get_params(self):
            return { 'policy': self.policy.state_dict(), 'value': self.value.state_dict() }

    def set_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.policy_old.load_state_dict(params['policy'])
        self.value.load_state_dict(params['value'])


# In[ ]:


import numpy as np
import torch

# -----------------------------
# Scalarization
# -----------------------------
def chebyshev_scalarization(obj, w, z_star):
    obj = np.array(obj)
    w = np.array(w)
    z_star = np.array(z_star)
    return np.max(w * np.abs(obj - z_star))

# -----------------------------
# Neighbor selection
# -----------------------------
def get_neighbors_indices(weights, T=2):
    P = len(weights)
    neighbors = []
    for p in range(P):
        distances = np.array([np.linalg.norm(weights[p] - weights[q]) if q != p else np.inf
                              for q in range(P)])
        neighbor_indices = distances.argsort()[:T]  # T closest
        neighbors.append(neighbor_indices.tolist())
    return neighbors

# -----------------------------
# KL divergence
# -----------------------------
def compute_kl_divergence(policy_p, policy_q, states, device='cpu'):
    with torch.no_grad():
        logits_p = policy_p(states.to(device))  # [N, action_dim]
        logits_q = policy_q(states.to(device))  # [N, action_dim]

        dist_p = torch.distributions.Categorical(logits=logits_p)
        dist_q = torch.distributions.Categorical(logits=logits_q)

        kl = torch.distributions.kl.kl_divergence(dist_p, dist_q)  # [N]
        return kl.mean().item()

# -----------------------------
# Adaptive tau
# -----------------------------
def adaptive_tau(kl_value, tau_max=0.5, lambda_kl=0.5):
    return tau_max * np.exp(-lambda_kl * kl_value)

# -----------------------------
# Dynamic ideal point
# -----------------------------
def get_dynamic_z_star(policies):
    objs = np.array([p.last_obj_rewards for p in policies])
    return objs.max(axis=0)

# -----------------------------
# MPCEM co-evolution
# -----------------------------
def multi_policy_coevolution(policies, z_star, tau_max=0.5, lambda_kl=0.5, T=2, device='cpu'):
    weight_vectors = [agent.w_vector.cpu().numpy() for agent in policies]
    neighbors_indices = get_neighbors_indices(weight_vectors, T)
    updated_policies = []

    # Compute dynamic ideal point
#    z_star = get_dynamic_z_star(policies)

    for p, policy in enumerate(policies):
        obj_p = policy.last_obj_rewards
        scalar_p = chebyshev_scalarization(obj_p, weight_vectors[p], z_star)
        theta_p = policy.get_params()  # current params

        # Sample states from replay memory for KL
        if len(policy.memory) > 0:
            states = np.array([t.state for t in policy.memory])
            states_sample = torch.tensor(states, dtype=torch.float32, device=device)
        else:
            states_sample = torch.zeros((1, policy.state_dim), dtype=torch.float32, device=device)

        for q in neighbors_indices[p]:
            neighbor = policies[q]
            obj_q = neighbor.last_obj_rewards
            scalar_q = chebyshev_scalarization(obj_q, weight_vectors[p], z_star)

            if scalar_q <= scalar_p:  # neighbor better or equal
                theta_q = neighbor.get_params()

                # KL divergence
                kl_value = compute_kl_divergence(policy.policy, neighbor.policy, states_sample, device)
                tau = adaptive_tau(kl_value, tau_max=tau_max, lambda_kl=lambda_kl)

                # Soft update per parameter
                for key in theta_p:
                    for k in theta_p[key]:
                        theta_p[key][k] = tau * theta_q[key][k] + (1 - tau) * theta_p[key][k]

        policy.set_params(theta_p)
        updated_policies.append(policy)

    return updated_policies


# In[ ]:


def train_mpppo(env, epochs=100, weight_vectors=None, episodes_per_epoch=5,
                gamma=0.99, eps_clip=0.2, lr_policy=1e-4, lr_value=1e-4,
                batch_size=1000, device='cpu', base_log_dir="./MPPOlogs", K=4,
                entropy_coef=0.01, ma_alpha=0.1, tau=0.05, alpha=1, beta=1,
                gamma_reward=1, C=10):

    import os
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    os.makedirs(base_log_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))]
    run_id = len(existing_runs) + 1
    log_dir = os.path.join(base_log_dir, f"MPPPO_{run_id}")
    os.makedirs(log_dir, exist_ok=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if weight_vectors is None:
        weight_vectors = [[1/3, 1/3, 1/3]]

    agents = []
    for w in weight_vectors:
        agent = MPPPOAgent(obs_dim, act_dim, weight_vector=w, gamma=gamma,
                           eps_clip=eps_clip, lr_policy=lr_policy, lr_value=lr_value)
        agent.last_obj_rewards = np.zeros(3, dtype=np.float32)
        agents.append(agent)

    writer = SummaryWriter(log_dir=log_dir)
    ma_rewards = [0.0 for _ in agents]

    for epoch in range(1, epochs + 1):
        for pid, agent in enumerate(agents):

            total_policy_loss = 0.0
            total_value_loss  = 0.0
            total_entropy     = 0.0

            epoch_obj_rewards = np.zeros(3, dtype=np.float32)
            epoch_steps = 0

            for ep in range(episodes_per_epoch):
                obs, _ = env.reset()
                done = False
                ep_reward = 0.0

                while not done:
                    obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

                    # ---- MASK HERE ----
                    action_mask = env.get_action_mask()  # np.array([1,1,0]) or [0,0,1]
                    action, log_prob = agent.select_action(obs, action_mask=action_mask, deterministic=False)

                    next_obs, reward_vector, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    # log realized objectives only on arrival
                    if info.get("event_type") == "arrival":
                        epoch_obj_rewards += np.asarray(info["realized_reward"], dtype=np.float32)
                        epoch_steps += 1

                    # (optional) scalar reward just for printing / MA
                    # NOTE: PPO still learns from reward_vector + agent.weight_vector unless you change scalarize_reward
                    scalar_reward = alpha * reward_vector[0] + beta * reward_vector[1] + gamma_reward * reward_vector[2]
                    scalar_reward = float(np.clip(scalar_reward, -10.0, 10.0))
                    ep_reward += scalar_reward

                    # ---- store transition ----
                    # If your memory doesn't store mask yet, remove action_mask from args
                    agent.store_transition(obs, action, reward_vector, next_obs, done, log_prob)  # OR add action_mask

                    obs = next_obs

                metrics = agent.update(
                    K=K,
                    batch_size=batch_size,
                    clip_grad_norm=0.5,
                    current_epoch=epoch,
                    total_epochs=epochs,
                    gae_lambda=0.95
                )

                total_policy_loss += metrics['policy_loss']  # DO NOT subtract entropy again (already in update)
                total_value_loss  += metrics['value_loss']
                total_entropy     += metrics['entropy']

                ma_rewards[pid] = ma_alpha * ep_reward + (1.0 - ma_alpha) * ma_rewards[pid]
                print(f"Epoch {epoch} | Policy {pid} | Episode {ep} | Ep Reward: {ep_reward:.3f} | MA Reward: {ma_rewards[pid]:.3f}")

            # average per-objective rewards (arrival only)
            avg_obj_rewards = epoch_obj_rewards / max(epoch_steps, 1)
            agent.last_obj_rewards = avg_obj_rewards

            policy_loss_mean = total_policy_loss / max(1, episodes_per_epoch)
            value_loss_mean  = total_value_loss  / max(1, episodes_per_epoch)
            entropy_mean     = total_entropy     / max(1, episodes_per_epoch)

            avg_Ut, avg_Dt, avg_Bt = map(float, avg_obj_rewards)

            writer.add_scalar(f'Policy_{pid}/Policy_Loss_Mean', policy_loss_mean, epoch)
            writer.add_scalar(f'Policy_{pid}/Value_Loss_Mean',  value_loss_mean,  epoch)
            writer.add_scalar(f'Policy_{pid}/Entropy_Mean',     entropy_mean,     epoch)

            writer.add_scalar(f'Policy_{pid}/Reward_Ut', avg_Ut, epoch)
            writer.add_scalar(f'Policy_{pid}/Reward_Dt', avg_Dt, epoch)
            writer.add_scalar(f'Policy_{pid}/Reward_Bt', avg_Bt, epoch)

            R_total = alpha * avg_Ut + beta * avg_Dt + gamma_reward * avg_Bt
            writer.add_scalar(f'Policy_{pid}/Reward_Total', R_total, epoch)

        # MPCEM
        if epoch % C == 0:
            z_star = get_dynamic_z_star(agents)
            agents = multi_policy_coevolution(
                policies=agents,
                z_star=z_star,
                tau_max=0.5,
                lambda_kl=0.5,
                T=2,
                device=device
            )
            print(f"Epoch {epoch} | MPCEM applied with KL-adaptive tau | z*: {z_star}")

    print(f"Logs saved in: {log_dir}")
    return agents

