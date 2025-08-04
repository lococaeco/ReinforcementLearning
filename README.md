<div align="center">

This repo is currently unstable except for dqn. However, it is continuously being updated.


<img src="https://myhits.vercel.app/api/hit/https%3A%2F%2Fdeku.posstree.com?color=blue&label=hits&size=small" alt="hits" />


# RL-code

A simple implementation of **Deep Q-Network (DQN)** using **OpenAI Gym**.

---

## Contents
 
1. [Prepare](#prepare)  
2. [Code_Structure](#Code_Structure)  
3. [Reference](#reference)

---



## Prepare

```python
conda create -n atari python=3.10
conda activate atari
pip install -r requirements.txt
```

## Code_Structure
```python
RL-code/                    
├── dqn/
│   ├── agent.py
│   ├── model.py
│   └── utils.py
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```


# Train
python train.py --env Breakout-v5

# Eval
python eval.py --env Breakout-v5

