import numpy as np
from collections import Counter
from helpers import *
<<<<<<< HEAD

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
num_sentences = 100

# This also only serves testing purposes: it makes sure we only use
# the words that actually occur in the first num_sentences
words_fr = set("".join(french.split("\n")[:num_sentences]).split(" "))
words_en = set("".join(english.split("\n")[:num_sentences]).split(" "))

########
## Initialize the translation probabilities t(f | e)
# Should we initialise q s.t. we can use it to initialise q for model2?
# Check if this is a proper probability distribution. I.e., should we normalize?
t = Vividict()
for k in range(num_sentences):
	for i, f in enumerate(sentences_fr[k]):
		for e in sentences_en[k]:
			if t[f][e] == {}:
				t[f][e] = np.random.rand(1)[0] #Should we pick a lower random nr, because the sum of p is way over 1



########### 
## E-M algorithm
num_timesteps = 200         
for ts in range(num_timesteps):
<<<<<<< HEAD
        #print "Starting iteration %s" % ts5
=======
        #print "Starting iteration %s" % ts
>>>>>>> master

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

#print t["et"]['and'] # converges to 1
#print t["et"]['first'] # conversges to 0
#print sum(t["et"][e] for e in words_en)# check whether probabilities eventually add up to 1!

=======
from time import time
import cProfile
from copy import copy
from datetime import datetime

class IBM1:
	"""Implementation of the IBM 1 model

	In the code, the list of english sentences is called `EN`
	and the one with french sentences `FR`. A sentence is always
	named by a single capital: `E` for an English sentence and 
	and `F` for a french. Words in a sentence are then the corresonding
	lowercase characters `e` and `f`.

	Arguments:
		english: the plain text with English sentences
		french: the plain French sentences
		null: (optional) How many null words to use (Extension from Moore, 2004)
		start: (optional) only use sentences from `start` onwards
		limit: (optional) only use sentences up to `limit`
		name: (optional) A name for the model
		desc: (optional) a description
		out_dir: (optional) write all files in this directory
	"""

	def __init__(self, english, french, 
		num_null = 1.0, add_n = 0.0, add_n_voc_size = 60000,
		name="", desc="", start=0, limit=-1, out_dir="", dump_trans_probs=False):
		self.FR = text2sentences(french)[start:limit]
		self.EN = text2sentences(english)[start:limit]
		self.EN = map(add_null, self.EN)

		self.voc_fr = sentences2voc(self.FR)
		self.voc_en = sentences2voc(self.EN)
		print "Done splitting sentences"

		self.num_null = num_null
		self.add_n = add_n
		self.add_n_voc_size = add_n_voc_size

		self.name = name.replace(" ","-").lower()
		self.desc = desc
		self.out_dir = out_dir
		self.start = start
		self.limit = limit
		self.dump_trans_probs = dump_trans_probs

	def initialize(self, logfreq=500):
		"""Uniformly initializes the translation probabilities
		Note that the translation probabilities are unnormalized
		"""
		print "Initializing..."
		t = Counter() 
		for k, (E, F) in enumerate(zip(self.EN, self.FR)):
			if (k % logfreq) == 0:
				print "\t%sk sentences initialized" % str(k/1000.0).zfill(5)
			for f in F:
				for e in E:
					t[(f, e)] = 1.0
		return t

	def train(self, num_iter, t=None, logfreq=500):
		"""Train the IBM1 model
		Return:
			t: the translation probabilities
			likelihoods: the log-likelihood of the data after every iteration
		"""
		if t is None: 
			t = self.initialize(logfreq=logfreq)
		likelihoods = []
		counts_ef = Counter()
		counts_e  = Counter()

		for ts in range(num_iter):
			print("Start iteration %s" % ts)

			t0 = time()
			tprev = t0
			counts_ef.clear()
			counts_e.clear()

			# E-step
			for k, (E, F) in enumerate(zip(self.EN, self.FR)):
				if (k % logfreq) == 0:
					print "\t%sk sentences done (%s / %ss)" % (str(k/1000.0).zfill(5), round(time()-t0, 2), round(time()-tprev, 2))
					tprev = time()

				for f in F:
					delta_sum = sum([t[(f, e)] for e in E])
					for e in E:
						# delta(k, j, j) = p(A_i = j | e, f, m )
						delta = t[(f, e)] / delta_sum
						counts_ef[(e, f)] += delta
						counts_e[e] += delta

			# M-step
			print "\tE-step done, maximizing translation probabilities..."
			for f, e in t.keys():

				# New transition probabilities with add-n smooting
				t[(f, e)] = (counts_ef[(e, f)] + self.add_n) / (counts_e[e] + self.add_n * self.add_n_voc_size)
				
				# And: multiply the null-words
				if e == "NULL": 
					t[(f, e)] *= self.num_null

			print "\tE-M done. Calculating likelihoods..."
			
			# Log likelihood
			likelihood = 0
			for F, E in zip(self.FR, self.EN):
				likelihood += self.log_likelihood(F, E, t)
			likelihoods += [likelihood]

			print "\tLog-likelhood: %s" % round(likelihood, 2)
			print "Iteration %s done in %ss.\n" % (ts, round(time() - t0, 1))
			
			if self.dump_trans_probs:
				self.dump_t(self.out_dir + self.name+"-trans-probs-iter-"+str(ts)+".txt", t)
		
		self.t = t
		self.likelihoods = likelihoods

		return t, likelihoods


	def log_likelihood(self, F, E, t):
		"""Log-likelihood of a pair of a French and English sentence"""
		L = 0
		for f in F:
			L += np.log(sum([t[(f, e)] for e in E]))

		# For normalization, you could multiply (substract) by (1/ (l +1) )^m
		return L #- len(F) * np.log(len(E))

	def posterior(i, f, E, t):
		"""The probability of aligning f to E[i]
		Or symbolically:
		$p( a_j = i | f, e_i) = t(f | e_i) / \sum_{i=1}^l t(f | e_i)$
		"""
		numerator = t[(f, E[i])]
		denominator = sum([ t[(f, e)] for e in E ])    
		return numerator/denominator if numerator != 0.0 else 0.0


	def decode(F, E, t=None):
		"""Gets the Viterbi alignment for two aligned sentences
		If alignment of some French word with the NULL-word is most 
		probable, the French word remains unaligned.

		Returns:
		A list of tuples $(f_i, e_{a_i}, p)$ indicating that
		f_i is aligned to e_{a_i} with probability p
		"""
		if t == None: t = self.t
		alignment = []
		for i, f in enumerate(F):
			alignment_probs = [posterior(j, f, E, t) for j in range(len(E))]
			best = np.argmax(alignment_probs) 
			if best != 0: 
				alignment.append((i, best, max(alignment_probs)))
		return alignment


	def dump_t(self, filename, t=None):
		if t == None: t = self.t
		with open(filename, 'w') as outfile:
			for (f, e), p in t.items():
				outfile.write("%s %s %s\n" % (f, e, np.log(p)))

	def load_t(self, filename, update=True):
		with open(filename, "r") as infile:
			t_new = Counter()
			for l in infile:
				parts = l.replace("\n","").split(" ")
				t_new[(parts[0], parts[1])] = np.exp(float(parts[2]))    
			
			if update: self.t = t_new
			return t_new

	def save_model(self):
		self.dump_t(self.out_dir + self.name+"-transition-probs.txt")
		with open(self.out_dir + self.name+"-log.txt", "w") as outfile:
			outfile.write("\n\n****************************************\n")
			outfile.write("* EXPERIMENT "+self.name+"\n*\n")
			outfile.write("* "+self.desc+"\n****************************************\n*\n")
			outfile.write("* Stored: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n*\n")
			outfile.write("* Number of null words: " + str(self.num_null) + "\n")
			outfile.write("* Trained on %s sentences (from %s to %s)\n" % (len(self.FR), self.start, self.limit))
			outfile.write("* In %s iterations \n" % len(self.likelihoods))
			# outfile.write("* Transition probabilities stored in: " + self.name+"-transition-probs.txt\n*\n")
			outfile.write("* Log likelihood during training:\n")
			for i, l in enumerate(self.likelihoods):
				outfile.write("*    %s)  %s\n" % (str(i).zfill(2), str(l)))


if __name__ ==  "__main__":
	english = open('data/hansards.36.2.e').read()
	french = open('data/hansards.36.2.f').read()

	M = IBM1(english, french,
		start=0, limit=100, add_n = .00001,
		name="Test", desc="Dit is een test model.", 
		out_dir="results/")
	M.train(3, logfreq=1000)
	M.save_model()
	
>>>>>>> master
