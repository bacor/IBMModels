import numpy as np
from collections import Counter
from helpers import *
from time import time

class IBM1:
	"""Implementation of the IBM 1 model

	In the code, the list of english sentences is called `EN`
	and the one with french sentences `FR`. A sentence is always
	named by a single capital: `E` for an English sentence and 
	and `F` for a french. Words in a sentence are then the corresonding
	lowercase characters `e` and `f`.
	"""

	def __init__(self, english, french, limit=-1):
		self.FR = text2sentences(french)[:limit]
		self.EN = text2sentences(english)[:limit]

		self.voc_fr = sentences2voc(self.FR)
		self.voc_en = sentences2voc(self.EN)


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

	def train(self, num_iter):
		"""Train the IBM1 model
		Return:
			t: the translation probabilities
			likelihoods: the log-likelihood of the data after every iteration
		"""

		t = self.initialize()
		likelihoods = []
		counts_ef = Counter()
		counts_e  = Counter()

		for ts in range(num_iter):
			t0 = time()
			counts_ef.clear()
			counts_e.clear()

			# E-step
			for E, F in zip(self.EN, self.FR):
				for f in F:
					delta_sum = sum([t[(f, e)] for e in E])					
					for e in E:
						# delta(k, j, j) = p(A_i = j | e, f, m )
						delta = t[(f, e)] / delta_sum
						counts_ef[(e, f)] += delta
						counts_e[e] += delta

			# M-step
			for f in self.voc_fr:
				for e in self.voc_en:
					t[(f, e)] = counts_ef[(e, f)] / counts_e[e]

			# Log likelihood
			likelihood = 0
			for F, E in zip(self.FR, self.EN):
				likelihood += self.log_likelihood(F, E, t)
			likelihoods += [likelihood]

			# Remove zero-entries
			t += Counter()

			print("Iteration %s finished in %ss. Log-likelhood: %s" % (ts, round(time() - t0, 1), round(likelihood, 2)))

		return t, likelihoods


	def log_likelihood(self, F, E, t):
		"""Log-likelihood of a pair of a French and English sentence"""
		L = 0
		for f in F:
			L += np.log(sum([t[(f, e)] for e in E]))

		# For normalization, you could multiply (substract) by (1/ (l +1) )^m
		return L #- len(F) * np.log(len(E))


if __name__ ==  "__main__":
	english_file = open('data/sample.e', 'r') # test?
	english = english_file.read() # test?
	english_full = open('data/hansards.36.2.e').read()

	french_file = open('data/sample.f', 'r') # test?
	french = french_file.read() # test?
	french_full = open('data/hansards.36.2.f').read()

	M = IBM1(english_full, french_full, limit=40) # Wat doe je hier?
	print M
	print("Start training")
	t, likelihoods = M.train(5)
	print likelihoods

