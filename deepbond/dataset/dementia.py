from deepbond.dataset import DataSet
import re

class DementiaDataSet(DataSet):
	def _clean_text_file(self, text):
		text = re.sub(r'(\S+)([\*\.])', r'\1 \2', text.strip())
		text = re.sub(r'([\*\.])(\S+)', r'\1 \2', text.strip())
		return text.strip()

class DementiaPUCDataSet(DataSet):
	def _clean_text_file(self, text):
		return self._normalize(re.sub(r'(\ +)|(\-)', ' ', text).strip())

	def _normalize(self, text):
		# text = re.sub(r'\baquia\b', 'aqui', text)
		# text = re.sub(r'\bpreh\b', 'pro', text)
		# text = re.sub(r'\bcasin\b', 'casinha', text)
		# text = re.sub(r'\baluma\b', 'alguma', text)
		# text = re.sub(r'\bcaz\b', 'casa', text)
		# text = re.sub(r'\bachno\b', 'acho', text)
		# text = re.sub(r'\bnarmazem\b', 'armazém', text)
		# text = re.sub(r'\bnarmazém\b', 'armazém', text)
		# text = re.sub(r'\bcachorrin\b', 'cachorrinho', text)
		# text = re.sub(r'\bcacho\b', 'cachorrinho', text)
		# text = re.sub(r'\bcach\b', 'cachorrinho', text)
		# text = re.sub(r'\bperdideu\b', 'perdido', text)
		# text = re.sub(r'\bdel\b', 'dele', text)
		# text = re.sub(r'\bcoiz\b', 'coisa', text)
		# text = re.sub(r'\broupeir\b', 'roupeiro', text)
		# text = re.sub(r'\belela\b', 'ela', text)
		return text
