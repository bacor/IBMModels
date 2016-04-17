# You might not want to run this as a script, since loading the model takes
# some time. In that case, run python in a shell, then you can run commands
# successively and play around with your model. In that case, you should copy
# past parts of this code in your shell.


## A demo
import numpy as np
from IBM1 import *
from IBM2 import *
from helpers import *

# Read out the data
english = open('data/hansards.36.2.e').read()
french = open('data/hansards.36.2.f').read()

# Make a model object
model = IBM1(english, french, 
	 # Make sure these parameters match the ones in the log file
	add_n = 0.0,
	num_null = 1.0,
	name="Test", desc="Dit is een test model.")

# Change this path
trans_probs_file = "results/params/2-ibm1-default-trans-probs-iter-5.txt" 

# Now load the transition probabilites, this takes a while...
print 'Loading model...'
model.load_t(trans_probs_file)
print 'The model has been loaded!'

# Now load some test data
test_english = open("data/test/test.e").read()
test_EN = text2sentences(test_english)
test_EN = map(add_null, test_EN)
test_french = open("data/test/test.f").read()
test_FR = text2sentences(test_french)

# Now for example inspect the decoding:
F = test_FR[1]
E = test_EN[1]
decoding = model.decode(F, E)
print model.show_decoding(decoding, F, E)

# Or show many decodings
for k, (F, E) in enumerate(zip(test_FR, test_EN)):
    if k < 10:
        decoding = model.decode(F, E)
        print model.show_decoding(decoding,F,E)

#############
# You can also write the decodings to a file 
# that you can later evaluate by running
# perl data/eval/wa_eval_align.pl data/answers/test.wa.nonullalign ibm1-decodings.txt
# where the file locations should of course be correct

# filename = "ibm1-decodings.txt"
# with open(filename, "w") as f:
#     for k, (F, E) in enumerate(zip(test_FR, test_EN)):
#         decoding = model.decode(F, E)
#         for i, a_i, p in decoding:
#             f.write("%s %s %s %s\n" % (str(k+1).zfill(4), i+1, a_i, 'P'))