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

deterministicTransitionProbability = [
    # transition probability is, given the state and action, what happens next?
    # level 0: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired" 
            # attack -> die, win
        [
            [1, 0],
            # rest -> die, rested
            [0, 1]
        ],
        # "rested"
        [
            # attack -> die, win
            [1, 0],
            # defend -> tired, rested
            [1, 0],
            # train -> die, next level and tired
            [0, 1]
        ]
    ],
    # level 1: tired = 0 rested = 1 dead = 2 win = 3
        [   
        # "tired"
            # attack -> die, win
        [
            [1, 0],
            # rest -> die, rested
            [0, 1]
        ],
        # "rested"
        [
            # attack -> die, win
            [1, 0],
            # defend -> tired, rested
            [1, 0],
            # train -> die, next level and tired
            [0, 1]
        ]
    ],
    # level 2: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired"
            # attack -> die, win
        [
            [1, 0],
            # rest -> die, rested
            [0, 1]
        ],
        # "rested"
        [
            # attack -> die, win
            [1, 0],
            # defend -> tired, rested
            [1, 0],
            # train -> die, next level and tired
            [0, 1]
        ]
    ],
        # level 3: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired"
            # attack -> die, win
        [
            [1, 0],
            # rest -> die, rested
            [0, 1]
        ],
        # "rested"
        [
            # attack -> die, win
            [1, 0],
            # defend -> tired, rested
            [1, 0],
            # train -> die, next level and tired
            [0, 1]
        ]
    ],
        # level 4: tired = 0 rested = 1 dead = 2 win = 3
    [   
        # "tired"
            # attack -> die, win
        [
            [1, 0],
            # rest -> die, rested
            [0, 1]
        ],
        # "rested"
        [
            # attack -> die, win
            [0, 1],
            # defend -> tired, rested
            [1, 0],
            # train -> die, next level and tired
            [0, 1]
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

actionValues = [
    [[0, 0], [0, 0, 0]],
    [[0, 0], [0, 0, 0]],
    [[0, 0], [0, 0, 0]],
    [[0, 0], [0, 0, 0]],
    [[0, 0], [0, 0, 0]]
]

def transit(level, status, action, p):
    newState = np.random.choice(transition[level][status][action], p=p[level][status][action]) 
    return newState

def mdp(policy, episodes, steps, exploration_rate, costOfLiving, learningRate, p, discountFactor):
    global stepCount

    for episode in range(int(episodes)): 
        # initialization
        level = L_0
        status = RESTED
        totalRewards = 0
        stepCount = 0

        for step in range(steps):
            # choose action
            # Choose action based on epsilon-greedy policy
            if np.random.rand() < exploration_rate:
                action = np.random.choice(actions[level][status], p=policy[level][status])
            else:
                action = np.argmax(actionValues[level][status][:]) # input is all 4 actions, return the greatest.
            # execute
            newState = transit(level, status, action, p)
            reward = newState['reward'] - costOfLiving
            newStatus = newState['status']
            newLevel = newState['level']
            terminal = newState['terminal'] 
            # print ("\n", actionValues[0])
            # print (actionValues[1])
            # print (actionValues[2])
            # print (actionValues[3])
            # print (actionValues[4], "\n")
            actionValues[level][status][action] += learningRate * (reward + discountFactor * np.max(actionValues[newLevel][newStatus][:]) - actionValues[level][status][action])
            totalRewards += reward

            level = newLevel
            status = newStatus

            if (terminal != None):
                break
                
            # check if reached a terminal state
            stepCount += 1

        rewards.append(totalRewards)
        stepCounts.append(stepCount)

episodes = 1/2*10000
steps = 10
# episodes = 10
# steps = 10
exploration_rate = 0.0001
costOfLiving = 0.01
learningRate = 0.01
rewards = []
stepCounts = []
discountFactor = 0.99
#deterministicTransitionProbability
mdp(randomPolicy, episodes, steps, exploration_rate, costOfLiving, learningRate, deterministicTransitionProbability, discountFactor)

print (actionValues[0])
print (actionValues[1])
print (actionValues[2])
print (actionValues[3])
print (actionValues[4])

# #print (randomPolicy)
# #p.pprint(initialPolicy)
# #p.pprint(vTable)
# #p.pprint(actionValues)

plt.plot(stepCounts, label='Total steps', color='green')
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Steps')
plt.show()
plt.legend()

plt.plot(rewards, label='Total Reward')
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
plt.legend()

input('press any key to exit')
