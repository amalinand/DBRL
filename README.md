# Adaptive Double-Booking Strategy for Outpatient Scheduling Using Multi-Objective Reinforcement Learning

**Ninda Nurseha Amalina**, Heungjo An  
Department of Industrial Engineering  
Kumoh National Institute of Technology, Gumi City, South Korea  

---

## Abstract

Patient no-shows disrupt outpatient clinic operations, reduce productivity, and may delay necessary care. Clinics often adopt overbooking or double-booking to mitigate these effects. However, poorly calibrated policies can increase congestion and waiting times. Most existing methods rely on fixed heuristics and fail to adapt to real-time scheduling conditions or patient-specific no-show risk.

To address these limitations, we propose an **adaptive outpatient double-booking framework** that integrates individualized no-show prediction with **multi-objective reinforcement learning**. The scheduling problem is formulated as a **Markov decision process**, and patient-level no-show probabilities estimated by a **Multi-Head Attention Soft Random Forest** model are incorporated into the reinforcement learning state.

We develop a **Multi-Policy Proximal Policy Optimization (MP-PPO)** method equipped with a **Multi-Policy Co-Evolution Mechanism**. Under this mechanism, we propose a novel **τ-rule** based on *Kullback–Leibler divergence*, enabling selective knowledge transfer among behaviorally similar policies. This improves convergence and expands the diversity of trade-offs.

In addition, **SHapley Additive exPlanations (SHAP)** is used to interpret both the predicted no-show risk and the agent’s scheduling decisions.

The proposed framework determines when to **single-book**, **double-book**, or **reject** appointment requests, providing a dynamic and data-driven alternative to conventional outpatient scheduling policies.

---

## Project Structure

├── Train.py # Training script for reinforcement learning agent
├── RL_model.py # Model architecture
├── data/ # (Not included) Private outpatient dataset
├── weights/ # (Not included) Trained model weights (.pth)
└── README.md # Project description

## Notes

⚠️ The outpatient dataset and trained model weight files are not included in this repository due to privacy and file size restrictions.
