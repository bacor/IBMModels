import numpy as np
from collections import Counter
from helpers import *


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
		print "Start initialization"
		# Counters are not made for storing floats, 
		# but indexing is so convenient...
		t = Counter() 

		# Random initialization
		for E, F in zip(self.EN, self.FR):
			for f in F:
				for e in E:
					if t[(f, e)] == 0:
						t[(f, e)] = np.random.rand(1)[0]

		print "Start normalization"

		# Normalize all 'rows' s.t. sum_fi t(f | e) = 1
		for e in self.voc_en:
			C = sum([ t[(f, e)] for f in self.voc_fr ])
			for f in self.voc_fr:
				t[(f, e)] = t[(f, e)] / C

		return t

	def train(self, num_iter):
		t = self.initialize()
		likelihoods = []

		for ts in range(num_iter):
			print "Starting iteration %s" % ts

			counts_ef = Counter()
			counts_e  = Counter()

			# E-step
			for E, F in zip(self.EN, self.FR):
				for f in F:
					# Outside the loop over english words since you only
					# need to calculate this once.
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

		return t, likelihoods


	def log_likelihood(self, F, E, t):
		L = 0
		for f in F:
			L += np.log(sum([t[(f, e)] for e in E]))

		# "Multiply" by (1/ (l +1) )^m and return
		return L - len(F) * np.log(len(E))


if __name__ ==  "__main__":
	english_file = open('data/sample.e', 'r')
	english = english_file.read()
	english_full = open('data/hansards.36.2.e').read()

	french_file = open('data/sample.f', 'r')
	french = french_file.read()
	french_full = open('data/hansards.36.2.f').read()

	M = IBM1(english_full, french_full, limit=1000)
	t, likelihoods = M.train(10)
	print likelihoods
