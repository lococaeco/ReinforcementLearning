<!-- <div align="center"> -->

This repo is currently unstable except for dqn. However, it is continuously being updated.


<table>
  <tr>
    <td align="center"><img src="img/Alien.gif" width="250"/><br>Alien</td>
    <td align="center"><img src="img/Breakout.gif" width="250"/><br>Breakout</td>
    <td align="center"><img src="img/Boxing.gif" width="250"/><br>Boxing</td>
  </tr>
</table>


<div align="center">
<table>
  <tr>
    <td align="center"><img src="img/Enduro.gif" width="250"/><br>Enduro</td>
    <td align="center"><img src="img/Pong.gif" width="250"/><br>Pong</td>
  </tr>
</table>
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


