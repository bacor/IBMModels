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
		name="", desc="", start=0, limit=-1, 
		out_dir="", dump_trans_probs=False, log=True):
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

	def initialize(self, logfreq=500, log=None, update=True):
		"""Uniformly initializes the translation probabilities
		Note that the translation probabilities are unnormalized
		"""
		if log == None: log = self.log
		if log: print "Initializing..."
		t = Counter() 
		for k, (E, F) in enumerate(zip(self.EN, self.FR)):
			if (k % logfreq) == 0 and log:
				print "\t%sk sentences initialized" % str(k/1000.0).zfill(5)
			for f in F:
				for e in E:
					t[(f, e)] = 1.0
		if update:
			self.t = t

		return t

	def train(self, num_iter, t=None, logfreq=500, log=None):
		"""Train the IBM1 model
		Return:
			t: the translation probabilities
			likelihoods: the log-likelihood of the data after every iteration
		"""
		if log == None: log = self.log
		if t is None: t = self.t
		likelihoods = []
		counts_ef = Counter()
		counts_e  = Counter()

		for ts in range(num_iter):
			if log: print("Start iteration %s" % ts)

			t0 = time()
			tprev = t0
			counts_ef.clear()
			counts_e.clear()

			# E-step
			for k, (E, F) in enumerate(zip(self.EN, self.FR)):
				if (k % logfreq) == 0 and log:
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
			if log: print "\tE-step done, maximizing translation probabilities..."
			for f, e in t.keys():

				# New transition probabilities with add-n smooting
				t[(f, e)] = (counts_ef[(e, f)] + self.add_n) / (counts_e[e] + self.add_n * self.add_n_voc_size)
				
				# And: multiply the null-words
				if e == "NULL": 
					t[(f, e)] *= self.num_null

			if log: print "\tE-M done. Calculating likelihoods..."
			
			# Log likelihood
			likelihood = 0
			for F, E in zip(self.FR, self.EN):
				likelihood += self.log_likelihood(F, E, t)
			likelihoods += [likelihood]

			if log: print "\tLog-likelhood: %s" % round(likelihood, 2)
			if log: print "Iteration %s done in %ss.\n" % (ts, round(time() - t0, 1))
			
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

	def posterior(self, i, f, E, t):
		"""The probability of aligning f to E[i]
		Or symbolically:
		$p( a_i = j | f, e_j) = t(f | e_j) / \sum_{j=1}^l t(f | e_j)$
		"""
		numerator = t[(f, E[i])]
		denominator = sum([ t[(f, e)] for e in E ])    
		return numerator/denominator if numerator != 0.0 else 0.0


	def decode(self, F, E, t=None):
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
			alignment_probs = [self.posterior(j, f, E, t) for j in range(len(E))]
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
			print "%s %s %s" % (i, (f+" ").ljust(span, '.'), e)
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
			return t_new

	def save_model(self):
		self.dump_t(self.out_dir + self.name+"-trans-probs.txt")
		with open(self.out_dir + self.name+"-log.txt", "w") as outfile:
			outfile.write("\n\n****************************************\n")
			outfile.write("* EXPERIMENT "+self.name+"\n*\n")
			outfile.write("* "+self.desc+"\n****************************************\n*\n")
			outfile.write("* Stored: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n*\n")
			outfile.write("* Number of null words: " + str(self.num_null) + "\n")
			outfile.write("* Add-n: " + str(self.add_n) + "\n")
			outfile.write("* Add-n vocabulary size: " + str(self.add_n_voc_size) + "\n")
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
		out_dir="results/", 
		log=True)
	# M.load_t("results/test-transition-probs.txt")
	M.initialize()
	M.train(5)
	# M.save_model()
	
	for k in range(7,20):
		decoding =  M.decode(M.FR[k], M.EN[k])
		M.show_decoding(decoding, M.FR[k], M.EN[k])
