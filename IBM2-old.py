# Imports
import numpy as np
from helpers import *

# Load sample data (first 100 lines only)
english_file = open('data/sample.e', 'r')
english = english_file.read()
french_file = open('data/sample.f', 'r')
french = french_file.read()

# Split the texts into lists of sentences, which in turn will be 
# lists of words:
# sentences_en [ ["Hello", "World", "!"], ["What", "a", "nice", "day"], ...]
sentences_fr = [s.split(" ") for s in french.split("\n")]
sentences_en = [s.split(" ") for s in english.split("\n")]


# This should be len(sentences_fr), but can be lowered in testing
num_sentences = 10

# This also only serves testing purposes: it makes sure we only use
# the words that actually occur in the first num_sentences
words_fr = set("".join(french.split("\n")[:num_sentences]).split(" "))
words_en = set("".join(english.split("\n")[:num_sentences]).split(" "))

########
## Initialize the translation probabilities t(f | e)
#
# Check if this is a proper probability distribution. I.e., should we normalize?
t = Vividict()

# parameter are initialised either 'uniform', 'random' or 'model1'
par_init = 'random'

for k in range(num_sentences):
    for i, f in enumerate(sentences_fr[k]):
        for e in sentences_en[k]:
            if t[f][e] == {}:
                if par_init == 'random':
                    t[f][e] = np.random.rand(1)[0]
                elif par_init == 'uniform':
                    t[f][e] = 1 / len(t[f])
                elif par_init == 'model1':
                    t[f][e] = 0,1
                                    



########### 
## E-M algorithm
num_timesteps = 20         
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
print t['et']
#print t["et"]['and'] # converges to 1
#print t["et"]['first'] # conversges to 0
#print sum(t["et"][e] for e in words_en)# check whether probabilities eventually add up to 1!

