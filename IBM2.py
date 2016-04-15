import numpy as np
from collections import Counter
from helpers import *
from time import time
import cProfile
from copy import copy
from datetime import datetime
from IBM1 import *
import timeit

class IBM2:
	"""Implementation of the IBM 2 model

	In the code, the list of english sentences is called `EN`
	and the one with french sentences `FR`. A sentence is always
	named by a single capital: `E` for an English sentence and 
	and `F` for a french. Words in a sentence are then the corresonding
	lowercase characters `e` and `f`.

	Arguments:
		english: the plain text with English sentences
		french: the plain French sentences
		num_null: (optional) How many null words to use (Extension from Moore, 2004)
		add_n: (optional) the add-n smooting parameter
		add_n_voc_size: (optinal) estimate of vocabulary size in add-n smoothing
		start: (optional) only use sentences from `start` onwards
		limit: (optional) only use sentences up to `limit`
		name: (optional) A name for the model
		desc: (optional) a description
		out_dir: (optional) write all files in this directory
		dump_trans_probs: (optional) store transition probabilities to file after every iteration
	"""

	def __init__(self, english, french, 
		num_null = 1.0, add_n = 0.0, add_n_voc_size = 60000,
		name="", desc="", start=0, limit=-1, out_dir="", dump_trans_probs=False, log=True):
		self.FR = text2sentences(french)[start:limit]
		self.EN = text2sentences(english)[start:limit]
		self.EN = map(add_null, self.EN)

		self.voc_fr = sentences2voc(self.FR)
		self.voc_en = sentences2voc(self.EN)
		if log: print "Done splitting sentences"

		self.num_null = num_null
		self.add_n = add_n
		self.add_n_voc_size = add_n_voc_size

		self.name = name.replace(" ","-").lower()
		self.desc = desc
		self.out_dir = out_dir
		self.start = start
		self.limit = limit
		self.dump_trans_probs = dump_trans_probs
		self.log = log

	def initialize(self, method="uniform", update=True, logfreq=500, log=None):
		"""Uniformly initializes the translation probabilities
		Note that the translation probabilities are unnormalized

		Args:
			method: (optional) either "uniform", "random" or the transition probabilities
					(a Counter object) from IBM1. Default: "uniform"
			update: (optional) Update the model's trans. and alignment probabilities with
					the new initialization?
			logfeq: (optional) Frequency of logging

		Returns:
			t: the transition probabilities
			q: the alignment probabilities
		"""
		if log == None: log = self.log
		if log: print "Initializing..."
		t = Counter() 
		q = Counter()

		if method != "uniform" and method != "random":
			if type(method) is str:
				t = self.load_t(method, update=False)
			else: 
				t = method
				

		for k, (E, F) in enumerate(zip(self.EN, self.FR)):
			if (k % logfreq) == 0 and log:
				print "\t%sk sentences initialized" % str(k/1000.0).zfill(5)

			for i, f in enumerate(F):
				for j, e in enumerate(E):
					if method == "uniform":
						t[(f, e)] = 1.0
						q[(j, i, len(E), len(F))] = 1.0

					elif method == "random":
						t[(f, e)] = np.random.rand(1)[0]
						q[(j, i, len(E), len(F))] = np.random.rand(1)[0]

					elif method == "ibm1":
						# t has already been set
						q[(j, i, len(E), len(F))] = 1.0 # Uniform
		
		if update:
			self.t = t
			self.q = q

		return t, q

	def train(self, num_iter, t=None, q=None, logfreq=500, log=None):
		"""Train the IBM1 model
		Args:
			t: (optional) the translation probabilities. Default: use own t
			q: (optional) alignment probabilities. Default: use own q
			
		Return:
			t: the translation probabilities
			q: alignment probabilities
			likelihoods: the log-likelihood of the data after every iteration
		"""
		if log is None: log = self.log
		if t is None: t = self.t
		if q is None: q = self.q

		likelihoods = []
		counts_ef = Counter()
		counts_e  = Counter()
		counts_jilm = Counter()
		counts_ilm = Counter()

		max_sentence_len_fr = max(map(len, self.FR))
		max_sentence_len_en = max(map(len, self.EN))

		for ts in range(num_iter):
			if log: print("Start iteration %s" % ts)

			t0 = time()
			tprev = t0
			counts_ef.clear()
			counts_e.clear()
			counts_jilm.clear()
			counts_ilm.clear()

			# E-step
			for k, (E, F) in enumerate(zip(self.EN, self.FR)):
				l, m = len(E), len(F)

				if (k % logfreq) == 0 and log:
					print "\t%sk sentences done (%s / %ss)" % (str(k/1000.0).zfill(5), round(time()-t0, 2), round(time()-tprev, 2))
					tprev = time()

				for i, f in enumerate(F):
					delta_sum = sum([t[(f, e)] * q[(j, i, l, m)] for j, e in enumerate(E)])
					for j, e in enumerate(E):
						delta = t[(f, e)] * q[(j, i, l, m)] / delta_sum
						counts_ef[(e, f)] += delta
						counts_e[e] += delta
						counts_jilm[(j, i, l, m)] += delta
						counts_ilm[(i, l, m)] += delta

			# M-step
			if log: print "\tE-step done, maximizing translation probabilities..."
			for f, e in t.keys():
				# New transition probabilities with add-n smooting
				t[(f, e)] = (counts_ef[(e, f)] + self.add_n) / (counts_e[e] + self.add_n * self.add_n_voc_size)
				
				# And: multiply the null-words
				if e == "NULL": 
					t[(f, e)] *= self.num_null

			if log: print "\tMaximizing alignment probabilities..."
			for j, i, l, m in q.keys():
				q[(j, i, l, m)] = counts_jilm[(j, i, l, m)] / counts_ilm[(i, l, m)]

			if log: print "\tE-M done. Calculating likelihoods..."
			likelihood = 0
			for F, E in zip(self.FR, self.EN):
				likelihood += self.log_likelihood(F, E, t, q)
			likelihoods += [likelihood]
			
			if log: print "\tLog-likelhood: %s" % round(likelihood, 2)
			if log: print "Iteration %s done in %ss.\n" % (ts, round(time() - t0, 1))
			if self.dump_trans_probs:
				self.dump_t(self.out_dir + self.name+"-trans-probs-iter-"+str(ts)+".txt", t)
		
		self.t = t
		self.q = q
		self.likelihoods = likelihoods

		return t, q, likelihoods

	def log_likelihood(self, F, E, t, q):
		"""Log-likelihood of a pair of a French and English sentence"""
		L = 0
		for i, f in enumerate(F):
			L += np.log(sum([t[(f, e)] * q[(j, i, len(E), len(F))] for j, e in enumerate(E)]))

		# For normalization, you could multiply (substract) by (1/ (l +1) )^m
		return L #- len(F) * np.log(len(E))

	def posterior(self, j, i, F, E, t=None, q=None):
		"""The probability of aligning f to E[i]
		Or symbolically:
		$p( a_i = j | f, e_j) = t(f | e_j) / \sum_{j=1}^l t(f | e_j)$
		"""
		if t == None: t = self.t
		if q == None: q = self.q
		f = F[i]
		numerator = t[(f, E[j])] * q[(j, i, len(E), len(F))]
		denominator = sum([ t[(f, e)] * q[(j, i, len(E), len(F))] for j, e in enumerate(E) ])
		if denominator > 0 :
			return numerator / denominator
		else:
			return 0
		# return 1
		# return numerator/denominator if numerator != 0.0 else 0.0

	def decode(self, F, E, t=None, q=None):
		"""Gets the Viterbi alignment for two aligned sentences
		If alignment of some French word with the NULL-word is most 
		probable, the French word remains unaligned.

		Returns:
		A list of tuples $(f_i, e_{a_i}, p)$ indicating that
		f_i is aligned to e_{a_i} with probability p
		"""
		if t == None: t = self.t
		if q == None: q = self.q
		alignment = []
		for i, f in enumerate(F):
			alignment_probs = [self.posterior(j, i, F, E, t, q) for j in range(len(E))]
			best = np.argmax(alignment_probs) 
			if best != 0: 
				alignment.append((i, best, max(alignment_probs)))
		return alignment

	def show_decoding(self, decoding, F, E):
		decode_dict = Counter()
		for i, j, p in decoding:
			decode_dict[i] = j
		
		print "".ljust(80, '-')
		print "French:  "+" ".join(F) + "\nEnglish: " + " ".join(E[1:])+"\n"

		span = max(map(len, F)) + 6
		for i, f in enumerate(F):
			e = E[decode_dict[i]]
			print "%s %s %s" % (i, f.ljust(span, '.'), e)
		print "".ljust(80, '-') + "\n"

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
			print "Translation probabilities loaded."
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
	# english = open('data/sample.e').read()
	# french = open('data/sample.f').read()

	# M1 = IBM1(english, french,
	# 	start=0, limit=5000, add_n=0,
	# 	name="test-ibm1", desc="Dit is een test model.", 
	# 	out_dir="results/", dump_trans_probs=False)
	# M1.load_t("results/test-ibm1-transition-probs.txt")
	# t1, l = M1.train(5)
	# M1.save_model()


	
	M = IBM2(english, french,
		start=0, limit=5000, add_n=0,
		name="Test", desc="Dit is een test model.", 
		out_dir="results/", log=True)
	M.initialize(method="random")
	M.train(1)
	# Uniform and random work fine. Loading IBM1 does not

	for k in range(7,20):
		decoding =  M.decode(M.FR[k], M.EN[k])
		M.show_decoding(decoding, M.FR[k], M.EN[k])



