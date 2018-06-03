import numpy as np

class Affixes:

	def __init__(self, start=1, end=4):
		self.start = start
		self.end = end
		self.mask = '_'

	@property
	def size(self):
		return self.end - self.start + 1

	def extract(self, word, which='prefix'):
		t = (self.end - self.start + 1)
		mins, maxs = self.start, self.end
		if self.start > len(word):
			mins = len(word)
		if self.end > len(word) or mins > self.end:
			maxs = len(word)
		a = []
		for i in range(mins, maxs + 1):
			w = word[:i] if which == 'prefix' else word[-i:]
			a.append(w)
		for _ in range(t - len(a)):
			a.append(self.mask)
		return a
