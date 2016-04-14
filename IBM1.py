import numpy as np
from collections import Counter
from helpers import *
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

	def __init__(self, english, french, num_null = 1.0, 
		name="", desc="", start=0, limit=-1, out_dir=""):
		self.FR = text2sentences(french)[start:limit]
		self.EN = text2sentences(english)[start:limit]
		self.EN = map(add_null, self.EN)

		self.voc_fr = sentences2voc(self.FR)
		self.voc_en = sentences2voc(self.EN)
		self.num_null = num_null

		self.name = name.replace(" ","-").lower()
		self.desc = desc
		self.out_dir = out_dir
		self.start = start
		self.limit = limit

	def initialize(self):
		"""Uniformly initializes the translation probabilities
		Note that the translation probabilities are unnormalized
		"""
		t = Counter() 
		for E, F in zip(self.EN, self.FR):
			for f in F:
				for e in E:
					t[(f, e)] = 1.0
		return t

	def train(self, num_iter, t=None, logfreq=100):
		"""Train the IBM1 model
		Return:
			t: the translation probabilities
			likelihoods: the log-likelihood of the data after every iteration
		"""
		if t is None: 
			t = self.initialize()
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
				t[(f, e)] = counts_ef[(e, f)] / counts_e[e]
				
				if e == "NULL": 
					# Multiple null words
					t[(f, e)] *= self.num_null

			print "\tE-M done. Calculating likelihoods..."
			
			# Log likelihood
			likelihood = 0
			for F, E in zip(self.FR, self.EN):
				likelihood += self.log_likelihood(F, E, t)
			likelihoods += [likelihood]

			print "\tLog-likelhood: %s" % round(likelihood, 2)
			print "Iteration %s done in %ss.\n" % (ts, round(time() - t0, 1))
			
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
			outfile.write("* Transition probabilities stored in: " + self.name+"-transition-probs.txt\n*\n")
			outfile.write("* Log likelihood during training:\n")
			for i, l in enumerate(self.likelihoods):
				outfile.write("*    %s)  %s\n" % (str(i).zfill(2), str(l)))


if __name__ ==  "__main__":
	english = open('data/hansards.36.2.e').read()
	french = open('data/hansards.36.2.f').read()

	M = IBM1(english, french, 
		start=0, limit=1000,
		name="Test", desc="Dit is een test model.", 
		out_dir="results/")
	M.train(3, logfreq=500)
	M.save_model()
	
