## A demo

import numpy as np
from IBM1 import *
from helpers import *

# Read out the data
english = open('data/hansards.36.2.e').read()
french = open('data/hansards.36.2.f').read()

# Make a model object
my_model = IBM1(english_full, french_full, 
	start=0, limit=1000, # it only looks at sentences start ... limit
	name="Test", desc="Dit is een test model.", 
	out_dir="results/") # Make a directory /outdir 

# Three iterations, log progress after every 500 sentences
t, likelihoods = my_model.train(3, logfreq=500) 

# Store the probabilities and some additional info
my_model.save_model()