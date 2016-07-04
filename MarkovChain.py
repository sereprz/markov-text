from collections import Counter
import numpy as np
import nltk
import re


class MarkovChain():
    def __init__(self, raw_text, order):

        self.order = order

        sentences = []
        for line in re.split(re.compile('\.|!|\?|,"'), raw_text):
            if line.strip() != '':
                sentences.append(line.strip())

        self.text = '$'.join([s[:1].lower() + s[1:] for s in sentences])

        tokens = ['$'] + nltk.word_tokenize(self.text)

        states = [' '.join(state) for state in nltk.ngrams(tokens, n=order)]
        states_freq = Counter(states)

        ngrams = [' '.join(ngram) for ngram in nltk.ngrams(tokens, n=order + 1)]
        ngrams_freq = Counter(ngrams)

        trans_prob = {}
        for ngram, freq in ngrams_freq.items():
            state = ' '.join(ngram.split(' ')[:-1])
            trans_prob[ngram] = float(freq)/states_freq[state]

        first = [state for state in states_freq if state.split(' ')[0] == '$']
        p_start = np.array([float(states_freq[state])/len(sentences) for state in first])

        self.first_state_prob = {'state': first, 'p': p_start}
        self.transitions = trans_prob

    def start_sentence(self):
        starts = self.first_state_prob['state']
        p = normalize(np.array(self.first_state_prob['p']))
        return np.random.choice(starts, p=p)

    def next_word(self, last_state):
        ngrams = [ngram for ngram in self.transitions
                  if ' '.join(ngram.split()[:-1]) == last_state]
        next = [ngram.split()[-1] for ngram in ngrams]
        p = normalize(np.array([self.transitions[ngram] for ngram in ngrams]))

        return np.random.choice(next, p=p)

    def generate_sentence(self):
        s = self.start_sentence()
        next = self.next_word(s)

        while next != '$':
            s += ' ' + next
            last = last_state(s, self.order)
            next = self.next_word(last)

        return ' '.join(s.split()[1:])


def normalize(p):
    if p.sum() < 1:
        p /= p.sum()
    return p


def last_state(sentence, order):
    word_list = sentence.split()
    return ' '.join(word_list[-order:])
