Verified on python 3.8.10 and packages used are gym, NumPy, random and math

# CartPole-QLearning
A free rotating joint connects a pole to a cart that runs along a friction-less track. A force of +1 or -1 is applied to the cart to move it right or left respectively. The goal is to keep the pole from falling over. Reinforcement Learning is used to balance the pole. Every time-step that the pole remains erect earns you a +1 reward. When the pole is more than 15 degrees from vertical or the cart goes more than 2.4 units away from the center, the episode terminates. Each episode is 200 frames long so the maximum reward is 200. Using Q-Learning based on MDP we try to maximize the rewards.

## Steps to clone the files:
```
cd <path_to_workspace>
git clone https://github.com/Rishabh96M/CartPole-QLearning.git
```

## Steps to run the files:
```
cd CartPole-QLearning/
python main.py
```
