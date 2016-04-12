
# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python/19829714#19829714

class Vividict(dict):
	"""Extension of normal dictionary class with hierarchical indexing

	Based on http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python
	"""

	def __missing__(self, key):
		value = self[key] = type(self)()
		return value

	def __add__(self, a):
		if self == {}:
			return a
	    
	def __iadd__(self, a):
		return a        