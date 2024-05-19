from core import GenerateMDP, GenerateRandomPolicy
from mdp import MDP, Test

import numpy as np


# MDP Settings
ATTACKED_CHANCE = 0.1
MAX_LEVEL = 5

mdp = GenerateMDP(MAX_LEVEL, ATTACKED_CHANCE)

state = mdp.reset()
totalRewards = 0

# Episode settings
EPISODES = 4000
STEPS = 100

rewards = []
stepCounts = []


def transit(level, status, action):
    newState = np.random.choice(transition[level][status][action], p=transitionProbability[level][status][action])
    return newState


def Simulate(policy: dict, mdp: MDP) -> None:
    global rewards, stepCounts

    for episode in range(EPISODES):
        totalRewards = 0
        stepCounts = 0

        state = mdp.reset()

        for step in range(STEPS):
            # choose action
            action = np.random.choice(mdp.getActions(), p=policy[state])

            # execute
            newState = transit(level, status, action)
            totalRewards += newState['reward']
            status = newState['status']
            level = newState['level']

            # check if reached a terminal state
            stepCounts += 1
            if newState['terminal'] is not None:
                break

        rewards.append(totalRewards)
        stepCounts.append(stepCounts)


Simulate(GenerateRandomPolicy(mdp), mdp)
