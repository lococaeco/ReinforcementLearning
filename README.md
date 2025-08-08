<!-- <div align="center"> -->

This repo is currently unstable except for dqn. However, it is continuously being updated.


<div style="display: flex; justify-content: center; gap: 10px;">

  <video src="img/Alien.mp4" width="250" controls></video>

  <video src="img/Breakout.mp4" width="250" controls></video>

  <video src="img/Boxing.mp4" width="250" controls></video>

</div>

# RL-code

A simple implementation of **Deep Reinforcement Learning Algorithms** using **OpenAI Gym**.

---

## Contents
 
1. [Prepare](#prepare)  
2. [Code_Structure](#Code_Structure)  
3. [Reference](#reference)

---

## Prepare

```bash
conda create -n atari python=3.10
conda activate atari
pip install -r requirements.txt
```

## Code_Structure
```plaintext
ReinforcementLearning/                    
├── DQN/
│   ├── DQN_train.py
│   ├── DQN_eval.py
├── REINFORCE/
├── ...
├── ...
├── requirements.txt
└── README.md
```

# Train
python ~/ReinforcementLearning/DQN/classic_dqn/DQN_train.py 

# Eval
python ~/ReinforcementLearning/DQN/classic_dqn/DQN_eval.py 


