from math import isclose
from random import choice, choices

import networkx as nx
import matplotlib.pyplot as plt


class MDP:
    def __init__(self, states, actions, transitions=None, rewards=None, gamma=0.9, eps=1e-6, T=100):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.eps = eps
        self.T = T

        self.currentState = states[0]
        self.stepsTaken = 0

    def reset(self):
        self.currentState = self.states[0]
        self.stepsTaken = 0

        return self.currentState

    @property
    def stateSpace(self) -> int:
        return len(states)

    @property
    def allActions(self) -> set:
        return set([a for actions in self.actions.values() for a in actions])

    @property
    def actionSpace(self) -> int:
        return len(self.allActions)  # To set to remove duplicate actions

    def getActions(self) -> list:
        return self.actions[self.currentState]

    def step(self, action):
        possibleOutcomes = self.transitions[self.currentState][action]  # {s: chance, s: chance, etc.}

        newState = choices(list(possibleOutcomes.keys()), possibleOutcomes.values())[0]
        reward = self.rewards[self.currentState][action][newState]

        self.currentState = newState
        self.stepsTaken += 1

        return (newState,
                reward,
                newState not in transitions,  # In our case we treat 'sinks' (states with no further transitions) as terminal.
                self.stepsTaken >= self.T,
                None)  # What should info be?

    def toGraph(self):
        G = nx.DiGraph()

        G.add_nodes_from(self.states, node_shape='s', color='blue')
        G.add_nodes_from(self.allActions, node_shape='o', color='yellow')

        # G.add_edges_from(self.actions)

        plt.show()


# Terminals + rested and tired (one per level)
states = [f"R-{i}" for i in range(1, 6)] + [f"R-{i}" for i in range(1, 6)] + ["Dead", "Won"]

actions = {f'R-{i}': ['attack', 'defend', 'train'] for i in range(1, 6)}
actions.update({f'T-{i}': ['attack', 'rest'] for i in range(1, 6)})

transitions = {f'R-{i}': {'attack': {'Won': -0.1 + 0.2 * i, 'Dead': 1.1 - 0.2 * i},
                          'defend': {f'T-{i}': 0.3, f'R-{i}': 0.7},  # Boss has a 30% chance to attack.
                          'train': {f'T-{i + 1}': 0.8, 'Dead': 0.2}
                          } for i in range(1, 6)}
transitions.update({f'T-{i}': {'rest': {f'R-{i}': 0.7, 'Dead': 0.3},
                               'attack': {'Won': -0.05 + 0.1 * i, 'Dead': 1.05 - 0.1 * i}} for i in range(1, 6)})

rewards = {f'R-{i}': {'attack': {'Won': 1, 'Dead': -1},
                      'defend': {f'T-{i}': 0, f'R-{i}': 0},  # Boss has a 30% chance to attack.
                      'train': {f'T-{i + 1}': 0, 'Dead': -1}
                      } for i in range(1, 6)}
rewards.update({f'T-{i}': {'rest': {f'R-{i}': 0, 'Dead': -1},
                           'attack': {'Won': 1, 'Dead': -1}} for i in range(1, 6)})

for s, a in transitions.items():
    for action, outcomes in a.items():
        assert action in a  # Making sure each action is legal
        assert isclose(sum(outcomes.values()), 1, abs_tol=1e-4)  # Making sure the sum of outcomes is effectively 1

mdp = MDP(states, actions, transitions=transitions, rewards=rewards)  # create an MDP


state = mdp.reset()  # reset/re-initialize
totalRewards = 0

for _ in range(5):
    action = choice(mdp.getActions())
    newState, reward, terminated, truncated, info = mdp.step(action)  # execute an action in the current state of MDP

    totalRewards += reward
    print(f"{state} -> {newState} via {action} (reward of {reward}, total of {totalRewards})")

    if terminated:
        print('Done!')
        break

    state = newState

