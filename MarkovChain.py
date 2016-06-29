from collections import Counter
import numpy as np
import nltk
import re


class MarkovChain():
    def __init__(self, raw_text):

        sentences = []
        for line in re.split(re.compile('\.|!|\?|"'), raw_text):
            if line.strip() != '':
                sentences.append(line.strip())

        self.text = '$'.join([s[:1].lower() + s[1:] for s in sentences])

        tokens = ['$'] + nltk.word_tokenize(self.text)
        states_freq = Counter(tokens)

        bigrams = list(nltk.bigrams(tokens))
        trans_freq = Counter(bigrams)

        trans_prob = {}
        for state, freq in trans_freq.items():
            trans_prob[state] = float(freq)/states_freq[state[0]]

        self.states_freq = states_freq
        self.trans_freq = trans_freq
        self.trans_prob = trans_prob

    def next_word(self, w):
        bgs = [bigram for bigram in self.trans_prob if bigram[0] == w]
        w2 = [bg[1] for bg in bgs]
        p = np.array([self.trans_prob[bg] for bg in bgs])
        if p.sum() < 1:
            p /= p.sum()

        return np.random.choice(w2, p=p)

    def generate_sentence(self):
        s = [self.next_word('$')]

        while s[-1] != '$':
            s.append(self.next_word(s[-1]))

        return ' '.join(s[:-1])
