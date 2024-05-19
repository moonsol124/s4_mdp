from itertools import product


import numpy as np

GAMMA = 0.9


def UpdateV(policy, state, probabilities, items, vTable) -> int:
    return sum(policy[state][i] * GetFutureReward(i, state, probabilities, items, vTable)
               for i in range(len(policy[state])))


# expected reward: ∑s′,rp(s′,r∣s,a)[r+γV(s′)], gamma is 0.9
def GetFutureReward(action, state, probabilities, items, vTable):
    item = items[state][action]
    return sum(probabilities[state][action][i] * (item[i]['reward'] + (GAMMA * vTable[item[i]['state']]))
               for i in range(len(probabilities[state][action])))


def ConvergeV(policy, states: list, probabilities, items, vTable) -> None:
    for i, j in product(range(100), range(len(states))):
        vTable[j] = UpdateV(policy, j, probabilities, items, vTable)


# policy improvement
def UpdatePolicy(policy, probabilities, items, vTable):
    # loop over states
    for i in range(len(probabilities)):
        # loop over actions
        q = [calculateQ(i, j, probabilities, items, vTable) for j in range(len(probabilities[i]))]

        policy[i] = {a: int(a == np.argmax(q)) for a in range(len(policy[i]))}

    return policy


# Q update function: Q(s,a)←Q(s,a)+α[r+γmax a′​Q(s′,a′)−Q(s,a)], gamma is 0.9
def calculateQ(state, action, probabilities, items, vTable):
    item = items[state][action]
    return sum(probabilities[state][action][i] * (item[i]['reward'] + (GAMMA * vTable[item[i]['state']]))
               for i in range(len(probabilities[state][action])))



def EvaluatePolicy(vTable: dict, n=100):
    # policy evaluation
    for _ in range(n):
        for s in vTable:
            vTable[s] = UpdateV(policy, s)


def GPI(iterations: int, policy: dict, states: list, probabilities, items, gamma: float):
    vTable = {s: 0 for s in states}

    for i in range(iterations):
        # policy evaluation
        ConvergeV(policy, states, probabilities, items, vTable, gamma)

        # policy improvement
        policy = UpdatePolicy(policy, probabilities, items, vTable, gamma)

    return policy

