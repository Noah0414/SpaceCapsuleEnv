from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
import random

class SpaceEnv(Env):
    def __init__(self):
        # Actions we can take: temperature down, no change in temperature, temperature up
        self.action_space = Discrete(3)
        # Temperature array with continuous values. Min temp = 0, Max temp = 100
        self.observation_space = Box(low=np.array([-200]), high=np.array([200]))

        # Set start temp with some random noise
        self.state = 38 + random.randint(-20,20)
        # Set shower length (in seconds)
        self.shower_length = 300
        
    def step(self, action):
        # Apply action. By default, action values are 0, 1, 2.
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward. If temperature is within optimal range, reward = 1.
        # Else, reward = -1
        if self.state >= -15 and self.state <= 15: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        self.state += random.randint(-1,1)
        # Set placeholder for info. This is just an OpenAI Gym requirement
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # If we want to visualize the sytem, we can implement the render function
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = 0 + random.randint(-20,20)
        # Reset shower time
        self.shower_length = 60 
        return self.state

env = SpaceEnv()

discreteDockingSpace = np.linspace(-200, 200, 100)

def getState(observation):
    discreteDocking = int(np.digitize(observation, discreteDockingSpace))

    return discreteDocking

numberActions = env.action_space.n
def maxAction(Q, state):    
    values = np.array([Q[state,a] for a in range(numberActions)])
    action = np.argmax(values)
    return action

# model hyperparameters
# ALPHA = learning rate, controls how fast the algorithm learns
ALPHA = 0.1
# GAMMA = discount factor for future rewards
GAMMA = 0.9
# EPS = epsilon parameter for the epsilon-greedy methods such as SARSA   
EPS = 1.0
#construct state space
states = []
for i in range(len(discreteDockingSpace)+1):
  states.append((i))

Q = {}
for s in states:
  for a in range(numberActions):
    Q[s, a] = 0

totalRewards = []
avg_Rewards = []
numGames = 5000

for i in range(numGames):
    if i % numGames == 0:
        print('starting game', i)
    observation = env.reset()
    s = getState(observation)
    rand = np.random.random()
    a = maxAction(Q, s) if rand < (1-EPS) else env.action_space.sample()
    done = False
    epRewards = 0
    while not done:
        observation_, reward, done, info = env.step(a)   
        s_ = getState(observation_)
        rand = np.random.random()
        a_ = maxAction(Q, s_) if rand < (1-EPS) else env.action_space.sample()
        epRewards += reward
        Q[s,a] = Q[s,a] + ALPHA*(reward + GAMMA*Q[s_,a_] - Q[s,a])
        s, a = s_, a_
    EPS -= 2/(numGames) if EPS > 0 else 0
    totalRewards.append(epRewards)
    avg_Rewards.append(np.mean(totalRewards[-10:]))
    #print('episode ', i, 'Total Rewards', totalRewards[-1],
                #'Average Rewards',avg_Rewards[-1],
                #'epsilon %.2f' % EPS)
   
plt.plot(avg_Rewards, 'b--')
plt.title('Average Rewards vs. Episodes')
plt.ylabel('Average Rewards')
plt.xlabel('Episodes')
plt.show() 