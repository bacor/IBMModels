# Imports
import numpy as np
from helpers import *

sentences_fr = [['das', 'haus'],['das', 'buch'],['ein', 'buch']]
sentences_en = [['the', 'house'],['the', 'book'],['a', 'book']]

words_fr = set(['das', 'haus', 'buch', 'ein'])
words_en = set(['the', 'house', 'book', 'a'])

num_sentences = 3

########
## Initialize the translation probabilities t(f | e)
#
# Check if this is a proper probability distribution. I.e., should we normalize?
t = Vividict()
for k in range(num_sentences):
	for i, f in enumerate(sentences_fr[k]):
		for e in sentences_en[k]:
			if t[f][e] == {}: # why a conditional?
				t[f][e] = np.random.rand(1)[0]

print t
########### 
## E-M algorithm
num_timesteps = 1000        
for ts in range(num_timesteps):
        #print "Starting iteration %s" % ts

        # Rest counts
        counts_ef = Vividict()
        counts_e  = Vividict()

        for k in range(num_sentences): 
                m = len(sentences_fr[k])
                l = len(sentences_en[k])
                for i, f in enumerate(sentences_fr[k]):

                        # Outside the loop over english words since you only
                        # need to calculate this once.
                        delta_sum = sum([t[f][e] for e in sentences_en[k]])

                        for j, e in enumerate(sentences_en[k]):

                                # Update all counts
                                delta = t[f][e] / delta_sum  
                                counts_ef[e][f] += delta
                                counts_e[e] += delta

        # Update the parameters t(f | e)
        for f in words_fr:
                for e in words_en:
                        t[f][e] = (counts_ef[e][f] + 0.0) / (counts_e[e] + 0.0)

print t
print sum(t['buch'][e] for e in words_en)

# {'buch': {'a': 0.00050143856526195963, 'house': 0.0, 'the': 5.6385343860565903e-301, 'book': 1.0}, 'ein': {'a': 0.99949856143473792, 'house': 0.0, 'the': 0.0, 'book': 7.1844716990123485e-300}, 'haus': {'a': 0.0, 'house': 0.99949797091102255, 'the': 2.0853225798067474e-299, 'book': 0.0}, 'das': {'a': 0.0, 'house': 0.00050202908897742414, 'the': 1.0, 'book': 6.3887476442035224e-301}}
# 1.00050143857
# Should 'house' be in the dictionary of 'buch'? There is no sentence pair (f,e) s.t. 'buch' is in f and ''house' is in e  
