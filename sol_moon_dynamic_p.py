from math import isclose
from random import choice, choices
import numpy as np
import matplotlib.pyplot as plt
import pprint as p

# 4 attacks
# 2 binary states * 5 levels = 10 states
# 4 attacks, so probability 40 states in total 
# when rested, can do 3 actions: defend, train, attack
# when tired, can do 2 actions: rest, attack
# as the level increases, the following changes:
# probability of winning and losing 

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

L_0 = 0
L_1 = 1
L_2 = 2
L_3 = 3
L_4 = 4
TIRED = 0
RESTED = 1
DEAD = 0
WIN = 1
TIRED_ATTACK = 0
TIRED_REST = 1
RESTED_ATTACK = 0
RESTED_DEFEND = 1
RESTED_TRAIN = 2
ZERO = 0
NEGATIVE_REWARD = -1
POSITIVE_REWARD = 1
SMALL_REWARD = 0.5

transition = [
    # transition probability is, given the state and action, what happens next?
    # level 0: tired = 0 rested = 1 dead = 2 win = 3
        [   
            # "tired"
                # attack -> die, win
            [
                [{"level": L_0, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_0, "status": TIRED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # rest -> die, rested
                [{"level": L_0, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_0, "status": RESTED, "terminal": None, "reward": ZERO}]
            ],
            # "rested"
            [
                # attack -> die, win
                [{"level": L_0, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_0, "status": RESTED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # defend -> tired, rested
                [{"level": L_0, "status": TIRED, "terminal": None, "reward": ZERO}, {"level": L_0, "status": RESTED, "terminal": None, "reward": ZERO}],
                # train -> die, next level and tired
                [{"level": L_0, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_1, "status": TIRED, "terminal": None, "reward": ZERO}]
            ]
        ],
    # level 1: tired = 0 rested = 1 dead = 2 win = 3
        [   
            # "tired"
                # attack -> die, win
            [
                [{"level": L_1, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_1, "status": TIRED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # rest -> die, rested
                [{"level": L_1, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_1, "status": RESTED, "terminal": None, "reward": ZERO}]
            ],
            # "rested"
            [
                # attack -> die, win
                [{"level": L_1, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_1, "status": RESTED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # defend -> tired, rested
                [{"level": L_1, "status": TIRED, "terminal": None, "reward": ZERO}, {"level": L_1, "status": RESTED, "terminal": None, "reward": ZERO}],
                # train -> die, next level and tired
                [{"level": L_1, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_2, "status": TIRED, "terminal": None, "reward": ZERO}]
            ]
        ],
    # level 2: tired = 0 rested = 1 dead = 2 win = 3
        [   
            # "tired"
                # attack -> die, win
            [
                [{"level": L_2, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_2, "status": TIRED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # rest -> die, rested
                [{"level": L_2, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_2, "status": RESTED, "terminal": None, "reward": ZERO}]
            ],
            # "rested"
            [
                # attack -> die, win
                [{"level": L_2, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_2, "status": RESTED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # defend -> tired, rested
                [{"level": L_2, "status": TIRED, "terminal": None, "reward": ZERO}, {"level": L_2, "status": RESTED, "terminal": None, "reward": ZERO}],
                # train -> die, next level and tired
                [{"level": L_2, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_3, "status": TIRED, "terminal": None, "reward": ZERO}]
            ]
        ],
                [   
            # "tired"
                # attack -> die, win
            [
                [{"level": L_3, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_3, "status": TIRED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # rest -> die, rested
                [{"level": L_3, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_3, "status": RESTED, "terminal": None, "reward": ZERO}]
            ],
            # "rested"
            [
                # attack -> die, win
                [{"level": L_3, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_3, "status": RESTED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # defend -> tired, rested
                [{"level": L_3, "status": TIRED, "terminal": None, "reward": ZERO}, {"level": L_3, "status": RESTED, "terminal": None, "reward": ZERO}],
                # train -> die, next level and tired
                [{"level": L_3, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_4, "status": TIRED, "terminal": None, "reward": ZERO}]
            ]
        ],
                [   
            # "tired"
                # attack -> die, win
            [
                [{"level": L_4, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_4, "status": TIRED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # rest -> die, rested
                [{"level": L_4, "status": TIRED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_4, "status": RESTED, "terminal": None, "reward": ZERO}]
            ],
            # "rested"
            [
                # attack -> die, win
                [{"level": L_4, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_4, "status": RESTED, "terminal": WIN, "reward": POSITIVE_REWARD}], 
                # defend -> tired, rested
                [{"level": L_4, "status": TIRED, "terminal": None, "reward": ZERO}, {"level": L_4, "status": RESTED, "terminal": None, "reward": ZERO}],
                # train -> die, next level and tired
                [{"level": L_4, "status": RESTED, "terminal": DEAD, "reward": NEGATIVE_REWARD}, {"level": L_4, "status": TIRED, "terminal": None, "reward": ZERO}]
            ]
        ],
]

transitionProbability = [
    # transition probability is, given the state and action, what happens next?
    # level 0: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired" 
            # attack -> die, win
        [
            [0.95, 0.05], 
            # rest -> die, rested
            [0.3, 0.7]
        ],
        # "rested"
        [
            # attack -> die, win
            [0.9, 0.1], 
            # defend -> tired, rested
            [0.3, 0.7],
            # train -> die, next level and tired
            [0.3, 0.7]
        ]
    ],
    # level 1: tired = 0 rested = 1 dead = 2 win = 3
        [   
        # "tired"
            # attack -> die, win
        [
            [0.85, 0.15], 
            # rest -> die, rested
            [0.3, 0.7]
        ],
        # "rested"
        [
            # attack -> die, win
            [0.7, 0.3], 
            # defend -> tired, rested
            [0.3, 0.7],
            # train -> die, next level and tired
            [0.3, 0.7]
        ]
    ],
    # level 2: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired"
            # attack -> die, win
        [
            [0.75, 0.25], 
            # rest -> die, rested
            [0.3, 0.7]
        ],
        # "rested"
        [
            # attack -> die, win
            [0.5, 0.5], 
            # defend -> tired, rested
            [0.3, 0.7],
            # train -> die, next level and tired
            [0.3, 0.7]
        ]
    ],
        # level 3: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired"
            # attack -> die, win
        [
            [0.65, 0.35], 
            # rest -> die, rested
            [0.3, 0.7]
        ],
        # "rested"
        [
            # attack -> die, win
            [0.3, 0.7], 
            # defend -> tired, rested
            [0.3, 0.7],
            # train -> die, next level and tired
            [0.3, 0.7]
        ]
    ],
        # level 4: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired"
            # attack -> die, win
        [
            [0.55, 0.45], 
            # rest -> die, rested
            [0.3, 0.7]
        ],
        # "rested"
        [
            # attack -> die, win
            [0.1, 0.9], 
            # defend -> tired, rested
            [0.3, 0.7],
            # train -> die, next level and tired
            [0.3, 0.7]
        ]
    ]
]

initialPolicy = [
    # level 0
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 1
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 2
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 3
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 4
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ]
]

randomPolicy = [
    # level 0
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 1
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 2
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 3
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ],
    # level 4
    [   
        # "tired"
            # attack -> die, win
        [
            0.5,
            # rest -> die, rested
            0.5
        ],
        # "rested"
        [
            # attack -> die, win
            1/3, 
            # defend -> tired, rested
            1/3,
            # train -> die, next level and tired
            1/3
        ]
    ]
]


actions = [
    # level 0
    [   
        # "tired"
            # attack -> die, win
        [
            0,
            # rest -> die, rested
            1
        ],
        # "rested"
        [
            # attack -> die, win
            0, 
            # defend -> tired, rested
            1,
            # train -> die, next level and tired
            2
        ]
    ],
    # level 1
    [   
        # "tired"
            # attack -> die, win
        [
            0,
            # rest -> die, rested
            1
        ],
        # "rested"
        [
            # attack -> die, win
            0, 
            # defend -> tired, rested
            1,
            # train -> die, next level and tired
            2
        ]
    ],
    # level 2
    [   
        # "tired"
            # attack -> die, win
        [
            0,
            # rest -> die, rested
            1
        ],
        # "rested"
        [
            # attack -> die, win
            0, 
            # defend -> tired, rested
            1,
            # train -> die, next level and tired
            2
        ]
    ],
    # level 3
    [   
        # "tired"
            # attack -> die, win
        [
            0,
            # rest -> die, rested
            1
        ],
        # "rested"
        [
            # attack -> die, win
            0, 
            # defend -> tired, rested
            1,
            # train -> die, next level and tired
            2
        ]
    ],
    # level 4
    [   
        # "tired"
            # attack -> die, win
        [
            0,
            # rest -> die, rested
            1
        ],
        # "rested"
        [
            # attack -> die, win
            0, 
            # defend -> tired, rested
            1,
            # train -> die, next level and tired
            2
        ]
    ]
]

vTable = [
    # status: tired rested
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]

def updateV(level, status):
    v = 0
    for i in range (len(initialPolicy[level][status])):
        policy = initialPolicy[level][status][i]
        expectedReward = getFutureReward(level, status, i)
        v = v + (policy * expectedReward)
    return v

def getFutureReward(level, status, action):
    r = 0

    for i in range(len(transition[level][status][action])):
        #         transition probability [current reward+gamma*V]
        curReward = transition[level][status][action][i]['reward']
        nextLevel = transition[level][status][action][i]['level']
        nextStatus = transition[level][status][action][i]['status']
        r = r + (transitionProbability[level][status][action][i] * (curReward+(0.9*vTable[nextLevel][nextStatus])))        
    return r

def policyImprovement(level, status):
    actions = []    
    # repeat as many times as the number of actions
    for i in range (len(initialPolicy[level][status])):
        q = 0
        expectedReward = calculateQ(level, status, i)
        q = expectedReward
        actions.append(q)
    
    return actions

def calculateQ(level, status, action):
    r = 0
    # repeats as many times as the number of transitions given action and state
    for i in range(len(transition[level][status][action])):
        #         transition probability [current reward+gamma*V]
        curReward = transition[level][status][action][i]['reward']
        nextLevel = transition[level][status][action][i]['level']
        nextStatus = transition[level][status][action][i]['status']
        r = r + (transitionProbability[level][status][action][i] * (curReward+(0.9*vTable[nextLevel][nextStatus])))
    return r

def policyEvaluation():
    for i in range(100):
        for i in range(len(vTable)):
            for j in range(len(vTable[i])):
                vTable[i][j] = updateV(i, j)

actionValues = [
    [[], []],
    [[], []],
    [[], []],
    [[], []],
    [[], []]
]

def updatePolicy():
    updatedPolicy = initialPolicy
    for level in range(len(initialPolicy)):
        for status in range(len(initialPolicy[level])):
            optimalAction = np.argmax(actionValues[level][status])
            #print(actionValues[level][status])
            #print(initialPolicy[level][status])
            for action in range(len(initialPolicy[level][status])):
                if (action == optimalAction):
                    updatedPolicy[level][status][action] = 1
                else:
                    updatedPolicy[level][status][action] = 0
    return updatedPolicy

# for level in range(len(initialPolicy)):
#     for status in range(len(initialPolicy[level])):
#         print (initialPolicy[level][status])
#         print (actionValues[level][status])

def GPI():
    global initialPolicy

    for i in range(1):
        # 100 sweeps are done per iteration for policy evaluation
        policyEvaluation()
        print (vTable[0])
        print (vTable[1])
        print (vTable[2])
        print (vTable[3])
        print (vTable[4])
            
        # get q values of the current policy after evaluation
        for level in range(len(vTable)):
            for status in range(len(vTable[level])):
                actionValues[level][status] = policyImprovement(level, status)
        
        print (actionValues[0])
        print (actionValues[1])
        print (actionValues[2])
        print (actionValues[3])
        print (actionValues[4])

        initialPolicy = updatePolicy()

GPI()
#p.pprint (randomPolicy)
#p.pprint (initialPolicy)

def transit(level, status, action):
    newState = np.random.choice(transition[level][status][action], p=transitionProbability[level][status][action]) 
    return newState

def mdp(policy, episodes, steps):
    global stepCount

    for episode in range(episodes): 
        # initialization
        level = L_0
        status = RESTED
        totalRewards = 0
        stepCount = 0
        for step in range(steps):
            # choose action
            action = np.random.choice(actions[level][status], p=policy[level][status])

            # execute
            newState = transit(level, status, action)
            totalRewards += newState['reward']
            status = newState['status']
            level = newState['level']

            # check if reached a terminal state
            stepCount += 1
            if (newState['terminal'] != None):
                break
        rewards.append(totalRewards)
        stepCounts.append(stepCount)
    
episodes = 4000
steps = 100

rewards = []
stepCounts = []
mdp(randomPolicy, episodes, steps)
#p.pprint(initialPolicy)
#p.pprint(vTable)
#p.pprint(actionValues)

# plt.plot(rewards, label='Total Reward')
# plt.title('Rewards per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.show()
# plt.legend()

# input('press any key to exit')



#p.pprint(transition[0][0])
#p.pprint(randomPolicy[0][0])

#p.pprint (vTable)
# p.pprint (actionValues)

#p.pprint(initialPolicy[0][0][0])
#p.pprint(initialPolicy) 
#p.pprint (values[0][0])
# p.pprint (vTable)
# print("q values")
#p.pprint (values)
#p.pprint (initialPolicy)

# loop over all states, calculate the Q of each of the actions and select the best one.



# print(calculateQ(0, 0, 0))
# print(calculateQ(0, 0, 1))
# print(calculateQ(0, 1, 0))
# print(calculateQ(0, 1, 1))
# print(calculateQ(0, 1, 2))


# 12 states in total, (5 levels)*(tired+rested)+(dead,win)

# rewards = {f'R-{i}': {'attack': {'Won': 1, 'Dead': -1},
#                       'defend': {f'T-{i}': 0, f'R-{i}': 0},  # Boss has a 30% chance to attack.
#                       'train': {f'T-{i + 1}': 0, 'Dead': -1}
#                       } for i in range(1, 6)}
# rewards.update({f'T-{i}': {'rest': {f'R-{i}': 0, 'Dead': -1},
#                            'attack': {'Won': 1, 'Dead': -1}} for i in range(1, 6)})

# for s, a in transitions.items():
#     for action, outcomes in a.items():
#         assert action in a  # Making sure each action is legal
#         assert isclose(sum(outcomes.values()), 1, abs_tol=1e-4)  # Making sure the sum of outcomes is effectively 1

# mdp = MDP(states, actions, transitions=transitions, rewards=rewards)  # create an MDP


# state = mdp.reset()  # reset/re-initialize
# totalRewards = 0

# for _ in range(5):
#     action = choice(mdp.getActions())
#     newState, reward, terminated, truncated, info = mdp.step(action)  # execute an action in the current state of MDP

#     totalRewards += reward
#     print(f"{state} -> {newState} via {action} (reward of {reward}, total of {totalRewards})")

#     if terminated:
#         print('Done!')
#         break

#     state = newState


        #p = transitionProbability[level][status][action][i]
        #v = vTable[nextLevel][nextStatus]
        #outcome = transitionProbability[level][status][action][i] * (curReward+(0.9*vTable[nextLevel][nextStatus]))
        #print (p, v, outcome, curReward)
        # r = r + (transitionProbability[level][status][action][i] * (curReward+(0.9*vTable[nextLevel][nextStatus])))
        