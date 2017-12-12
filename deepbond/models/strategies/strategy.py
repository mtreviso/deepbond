from abc import ABCMeta, abstractmethod

class Strategy(metaclass=ABCMeta):

	def __init__(self, name='bucket', input_length=None):
		self.name = name
		self.input_length = input_length

	@abstractmethod
	def prepare_input(self, sequence):
		pass

	@abstractmethod
	def unprepare(self, sequence, **kwargs):
		pass

	@abstractmethod
	def prepare_output(self, sequence, one_hot_dim=None):
		pass

	def save(self, filename):
		import json
		strategy_info = {
			'input_length': self.input_length,
			'name': self.name
		}
		with open(filename, 'w') as f:
			json.dump(strategy_info, f)

	def load(self, filename):
		import json
		with open(filename, 'r') as f:
			data = json.load(f)
		self.__init__(**data)
