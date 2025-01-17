
def text2sentences(text):
	"""Splits a long string into sentences, which are lists of words"""
	sentences = []
	for line in text.split("\n"):
		word_list = [word for word in line.split(" ") if word != ""]
		sentences.append(word_list)
	return sentences

def sentences2voc(sentences):
	"""Extracts the vocabulary from a set of sentences (lists of words)"""
	return list(set([word for s in sentences for word in s]))

def add_null(sentence):
	"""Adds a nullword to a sentence (list of words)"""
	return ["NULL"] + sentence
    
# NOT NEEDED ANYMORE; USE COUNTER INSTEAD
class Vividict(dict):
	"""Extension of normal dictionary class with hierarchical indexing

	Based on http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python

	This you can do with dictionaries
	```
	mydict['a'] = {} 				
	mydict['a']['b'] = 3
	mydict['a']['b'] # returns 3
	```
	But this is a bit annoying, so Vividict implements a kind of hierarchical indexing:
	```
	mydict = Vividict()
	mydict['a']['b'] = 3 # Works!
	```
	Also, it makes calculations with dictionaries a bit easier. Roughly,
	it will treat an empty dictionary as a zero when addition is applied to it.
	If `mydict['a']['b']` is numerical, we can of course calculate with it. So 
	in our earlier example `mydict['a']['b'] += 2` will increment `mydict['a']['b']`
	to `5`. But what if we increment `mydict['a']['c']`, when the key `c` does not exist?
	That's why it treats empty dicts as zeros. So now we can just do
	```
	mydict = Vividict()
	mydict['c'] += 3 # Works, even though the key `c` does not exist!
	```
	This can be pretty helpful :-)

	"""

	def __missing__(self, key):
		value = self[key] = type(self)()
		return value

	def __add__(self, a):
		if self == {}:
			return a
	    
	def __iadd__(self, a):
		return a        