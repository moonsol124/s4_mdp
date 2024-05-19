from mdp import ActionsFromTransitions, MDP

from frozendict import frozendict
from matplotlib import pyplot as plt
from pprint import pprint


class fdict(frozendict):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        return f"L{self['level']} ({'T' if self['tired'] else 'R'})"

    def __repr__(self):
        return self.__str__()


def ToTired(s: frozendict) -> frozendict:
    return fdict(level=s['level'],
                 tired=True)


def ToRested(s: frozendict) -> frozendict:
    return fdict(level=s['level'],
                 tired=False)


def ToTrained(s: frozendict) -> frozendict:
    return fdict(level=s['level'] + 1,
                 tired=True)


def WinChance(s: frozendict, maxLevel: int, invert: bool = False) -> float:
    winChance = 0.1 + (0.9 / (maxLevel - 1)) * (s['level'] - 1)

    if s['tired']:
        winChance /= 2

    return (1 - winChance) if invert else winChance


def GenerateMDP(maxLevel: int, attackedChance: float) -> MDP:
    states = [fdict(level=l, tired=t) for l in range(1, maxLevel + 1) for t in (True, False)]
    statesPlus = states + ["Won", "Died"]

    transitions = {}

    for s in states:
        t = {'attack': {'Won': WinChance(s, maxLevel),
                        'Died': WinChance(s, maxLevel, True)}}

        if s['tired']:
            t['rest'] = {ToRested(s): 1 - attackedChance,
                         'Died': attackedChance}
        else:
            t['defend'] = {s: 1 - attackedChance,
                           ToTired(s): attackedChance}

            if s['level'] < maxLevel:
                t['train'] = {ToTrained(s): 1 - attackedChance,
                              'Died': attackedChance}

        transitions[s] = t

    rewards = {s: {'attack': {'Won': 1, 'Dead': -1},
                   'defend': {'Died': -1},
                   'train': {'Died': -1},
                   'rest': {'Died': -1}}
               for s in states}

    pprint(ActionsFromTransitions(transitions))

    return MDP(statesPlus,
               ActionsFromTransitions(transitions),
               transitions=transitions,
               rewards=rewards,
               startingState=fdict(level=1, tired=False))


def GenerateRandomPolicy(mdp: MDP) -> dict:
    return {s: {a: 1 / len(actions) for a in actions}
            for s, actions in ActionsFromTransitions(mdp.transitions).items()}


def Plot(x: list) -> None:
    plt.plot(x, label='Total Reward')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
