from collections import Counter
from operator import itemgetter
import numpy as np


class MarkovChain():
    def __init__(self, text, order):

        self.text = text
        self.order = order

        tokens = ['<eos>'] + self.text.split()
        n_words = len(set(tokens))

        states = zip(*[tokens[i:] for i in range(order)])
        n_states = len(set(states))

        states_lookup = dict(zip(set(states), range(n_states)))
        words_lookup = dict(zip(sorted(set(tokens)), range(n_words)))

        counts = np.zeros((n_states, n_words))

        for ngram, c in Counter(zip(*[tokens[i:] for i in range(order + 1)])).iteritems():
            x = states_lookup[ngram[:order]]
            y = words_lookup[ngram[order]]
            counts[x, y] = c

        with np.errstate(invalid='ignore', divide='ignore'):
            P = counts/counts.sum(axis=1)[:, None]
            P[np.isnan(P)] = 0

        self.begin = sorted([state for state in set(states) if state[0] == '<eos>'],
                            key=itemgetter(1))

        self.vocab = sorted(words_lookup.keys())
        self.states = states_lookup
        self.P = P

    def start_sentence(self):
        return self.begin[np.random.choice(len(self.begin))]

    def next_word(self, last_state):
        row = self.states[last_state]
        return np.random.choice(self.vocab, p=self.P[row])

    def generate_sentence(self):
        last = self.start_sentence()
        s = ' '.join(last)
        next = self.next_word(last)

        while next != '<eos>':
            s += ' ' + next
            last = last_state(s, self.order)
            next = self.next_word(last)

        return ' '.join(s.split()[1:])


def last_state(sentence, order):
    word_list = sentence.split()
    return tuple(word_list[-order:])
