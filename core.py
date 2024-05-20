from mdp import ActionsFromTransitions, MDP, State

from matplotlib import pyplot as plt
from numpy.random import choice
from pprint import pprint


def ToTired(s: State) -> State:
    return State(level=s['level'],
                 tired=True)


def ToRested(s: State) -> State:
    return State(level=s['level'],
                 tired=False)


def ToTrained(s: State) -> State:
    return State(level=s['level'] + 1,
                 tired=True)


def WinChance(s: State, maxLevel: int, invert: bool = False) -> float:
    # winChance = (1 / maxLevel) * s['level']
    winChance = 0.1 + (0.9 / (maxLevel - 1)) * (s['level'] - 1)

    if s['tired']:
        winChance /= 2

    return (1 - winChance) if invert else winChance


def GenerateMDP(maxLevel: int, attackedChance: float, costOfLiving: float = 0) -> MDP:
    states = [State(level=l, tired=t) for l in range(1, maxLevel + 1) for t in (True, False)]

    won, died = State(title="Won"), State(title="Died")
    statesPlus = states + [won, died]

    transitions = {}

    for s in states:
        t = {'attack': {won: WinChance(s, maxLevel),
                        died: WinChance(s, maxLevel, True)}}

        if s['tired']:
            t['rest'] = {ToRested(s): 1 - attackedChance,
                         died: attackedChance}
        else:
            t['defend'] = {s: 1 - attackedChance,
                           ToTired(s): attackedChance}

            if s['level'] < maxLevel:
                t['train'] = {ToTrained(s): 1 - attackedChance,
                              died: attackedChance}

        transitions[s] = t

    rewards = {s: {'attack': {won: 1, died: -1},
                   'defend': {died: -1},
                   'train': {died: -1},
                   'rest': {died: -1}}
               for s in states}

    return MDP(states,
               statesPlus,
               ActionsFromTransitions(transitions),
               transitions=transitions,
               rewards=rewards,
               costOfLiving=costOfLiving,
               startingState=State(level=1, tired=False))


def GenerateRandomPolicy(mdp: MDP) -> dict:
    return {s: {a: 1 / len(actions) for a in actions}
            for s, actions in ActionsFromTransitions(mdp.transitions).items()}


def Plot(x: list, title: str) -> None:
    plt.plot(x, label='Total Reward')
    plt.title(f'Rewards per Episode ({title})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def Simulate(policy: dict, mdp: MDP, episodes: int, steps: int) -> tuple[list, list]:
    rewards = []
    stepCounts = []

    for episode in range(episodes):
        totalRewards = 0
        stepCount = 0

        state = mdp.reset()

        for step in range(steps):
            action = choice(list(policy[state].keys()),
                            p=list(policy[state].values()))

            state, reward, terminated, truncated, info = mdp.step(action, message=False)
            totalRewards += reward

            stepCount += 1
            if terminated:
                break

        rewards.append(totalRewards)
        stepCounts.append(stepCount)

    return rewards, stepCounts
