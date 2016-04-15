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
q = Vividict()
par_init = 'uniform'

for k in range(num_sentences):
    for i, f in enumerate(sentences_fr[k]):
        for j,e in enumerate(sentences_en[k]):
            t[f][e]
            q[j][(i,len(sentences_en[k]),len(sentences_fr[k]))] = np.random.rand(1)[0]
            

for f in t:
    for e in t[f]:
        if par_init == 'random':
            t[f][e] = np.random.rand(1)[0]
        elif par_init == 'uniform':
            t[f][e] = 1 / float(len(t[f]))
        elif par_init == 'model1':
            t[f][e] = 1 / float(len(t[f]))
                                   
<<<<<<< HEAD
#print t
#print q

########### 
## E-M algorithm
num_timesteps = 25       
=======
print t
print q

########### 
## E-M algorithm
num_timesteps = 25        
>>>>>>> master
for ts in range(num_timesteps):
        #print "Starting iteration %s" % ts

        # Rest counts
        counts_ef = Vividict()
        counts_e  = Vividict()
        counts_jilm = Vividict()
        counts_ilm = Vividict()

        for k in range(num_sentences): 
                m = len(sentences_fr[k])
                l = len(sentences_en[k])
                for i, f in enumerate(sentences_fr[k]):

                        # Outside the loop over english words since you only
                        # need to calculate this once.
<<<<<<< HEAD
                        delta_sum = sum(t[f][e] * q[j][(i,l,m)] for j,e in enumerate(sentences_en[k]))
=======
                        delta_sum = sum(t[f][e] * q[j][(i,l,m)] for j,e in enumerate(sentences_en[k])) # fixen
>>>>>>> master

                        for j, e in enumerate(sentences_en[k]):

                                # Update all counts
                                delta = t[f][e] / delta_sum  
                                counts_ef[e][f] += delta
                                counts_e[e] += delta
<<<<<<< HEAD
                                counts_jilm[j][(i,l,m)] += delta
                                counts_ilm[(i,l,m)] += delta
=======
                                counts_jilm += delta
                                counts_ilm += delta
>>>>>>> master

        # Update the parameters t(f | e)
        for f in words_fr:
                for e in words_en:
                        t[f][e] = (counts_ef[e][f] + 0.0) / (counts_e[e] + 0.0)
<<<<<<< HEAD
        print q[0][(0, 2, 2)]
        print q[0][(1,2,2)]
        
        #update the parameters q([j|i,l,m])
        for j in range(max(len(f) for f in sentences_fr)):
            for i in range(max(len(e) for e in sentences_en)):
                for l in range(1, max(len(e) for e in sentences_en) + 1):
                    for m in range(1, max(len(f) for f in sentences_fr) + 1):
                        #print (j,i,l,m)
                        if counts_ilm[(i,l,m)] != {}:
                            q[j][(i,l,m)] = (counts_jilm[j][(i,l,m)] + 0.0) / (counts_ilm[(i,l,m)] + 0.0)

#print t
# print sum(t[f]['book'] for f in words_fr)
=======

        #update the parameters q([j|i,l,m])
        

print t
#print sum(t['buch'][e] for e in words_en)
>>>>>>> master

# {'buch': {'a': 0.00050143856526195963, 'house': 0.0, 'the': 5.6385343860565903e-301, 'book': 1.0}, 'ein': {'a': 0.99949856143473792, 'house': 0.0, 'the': 0.0, 'book': 7.1844716990123485e-300}, 'haus': {'a': 0.0, 'house': 0.99949797091102255, 'the': 2.0853225798067474e-299, 'book': 0.0}, 'das': {'a': 0.0, 'house': 0.00050202908897742414, 'the': 1.0, 'book': 6.3887476442035224e-301}}
# 1.00050143857
