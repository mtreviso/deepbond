def _remove_correferences(self, verbose=False):
	'''
	leva em consideração anáforas e catáforas muito adiantes, por ex:
	atual: alinea a)
	proximo: alinea c) --> catáfora
	se for alínea b), então é considerado uma continuação
	'''
	matches = lambda r, s: [s[m.start(0):m.end(0)] for m in re.finditer(r, s)]
	clue_str = lambda s: s.strip().replace('[', '').replace(']', '').lower()
	regex, clues_sub = zip(*self.regex_str)
	regexes = dict(zip(list(map(clue_str, clues_sub)), regex))
	cardinals_to_index = dict(zip(cardinals, range(len(cardinals))))
	ordinals_to_index = dict(zip(ordinals, range(len(ordinals))))
	get_numeral_alpha = lambda x, s: ord(s.split(x)[1].strip())
	atuais = dict(zip(list(map(clue_str, clues_sub)), list(map(lambda x: clue_str(x)+' -', clues_sub))))
	
	def get_numeral(x, s):
		if 'único' in s:
			return -1
		if '-' in s:
			return 0
		t = s.split(x)[1].strip()
		if t in cardinals_to_index:
			return cardinals_to_index[t] + 1
		return ordinals_to_index[t] + 1
	
	constituicao = {}
	new_texts = []
	qtd_inciso, qtd_alinea, qtd_numero = 0, 0, 0
	for i in range(len(self.texts)):
		print(i+1)

		w = 'título'
		for s in matches(regexes[w], self.texts[i]):
			c = constituicao
			if get_numeral(w, atuais[w])+1 == get_numeral(w, s) or len(c) == 0:
				atuais[w] = s
				constituicao[s] = {}
				self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
			elif verbosese:
				print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))

		w = 'capítulo'
		for s in matches(regexes[w], self.texts[i]):
			c = constituicao[atuais['título']]
			if get_numeral(w, atuais[w])+1 == get_numeral(w, s) or len(c) == 0:
				atuais[w] = s
				c[s] = {}
				self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
			elif verbose:
				print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))

		w = 'artigo'
		for s in matches(regexes[w], self.texts[i]):
			if atuais['capítulo'] not in constituicao[atuais['título']]:
				constituicao[atuais['título']][atuais['capítulo']] = {}
			c = constituicao[atuais['título']][atuais['capítulo']]
			if get_numeral(w, atuais[w])+1 == get_numeral(w, s) or len(c) == 0:
				atuais[w] = s
				c[s] = {}
				self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
			elif verbose:
				print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))

		w = 'parágrafo'
		for s in matches(regexes[w], self.texts[i]):
			c = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']]
			if get_numeral(w, atuais[w])+1 == get_numeral(w, s) or len(c) == 0:
				atuais[w] = s
				c[s] = {}
				self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
			elif verbose:
				print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))

		w = 'inciso'
		for s in matches(regexes[w], self.texts[i]):
			try:
				c1 = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']]
				if atuais['parágrafo'] not in constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']]:
					constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']] = {}
				c2 = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']]
				if get_numeral(w, atuais[w])+1 == get_numeral(w, s) or sum(w in x for x in c1.keys()) == 0:
					atuais[w] = s
					c1[s] = {}
					self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
				elif get_numeral(w, atuais[w])+1 == get_numeral(w, s) or len(c2) == 0:
					atuais[w] = s
					c2[s] = {}
					self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
				elif verbose:
					print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))
			except:
				if verbose:
					print('REGEX DETECTOU ERRONEAMENTE UM INCISO!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))
					qtd_inciso += 1

		w = 'alínea'
		for s in matches(regexes[w], self.texts[i]):
			try:
				if atuais['inciso'] not in constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']]:
					constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['inciso']] = {}
				c1 = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['inciso']]
				if atuais['inciso'] not in constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']]:
					constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']][atuais['inciso']] = {}
				c2 = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']][atuais['inciso']]
				if get_numeral_alpha(w, atuais[w])+1 == get_numeral_alpha(w, s) or sum(w in x for x in c1.keys()) == 0:
					atuais[w] = s
					c1[s] = {}
					self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
				elif get_numeral_alpha(w, atuais[w])+1 == get_numeral_alpha(w, s) or len(c2) == 0:
					atuais[w] = s
					c2[s] = {}
					self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
				elif verbose:
					print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))
			except:
				if verbose:
					print('REGEX DETECTOU ERRONEAMENTE UMA ALÍNEA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))
					qtd_alinea += 1

		w = 'número'
		for s in matches(regexes[w], self.texts[i]):
			try:
				if atuais['alínea'] not in constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['inciso']]:
					constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['inciso']][atuais['alínea']] = {}
				c1 = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['inciso']][atuais['alínea']]
				if atuais['alínea'] not in constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']][atuais['inciso']]:
					constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']][atuais['inciso']][atuais['alínea']] = {}
				c2 = constituicao[atuais['título']][atuais['capítulo']][atuais['artigo']][atuais['parágrafo']][atuais['inciso']][atuais['alínea']]
				print(w, atuais[w], s)
				if get_numeral(w, atuais[w])+1 == get_numeral(w, s) or sum(w in x for x in c1.keys()) == 0:
					atuais[w] = s
					c1[s] = {}
					self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
				elif get_numeral(w, atuais[w])+1 == get_numeral(w, s) or len(c2) == 0:
					atuais[w] = s
					c2[s] = {}
					self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)
				elif verbose:
					print('CORREFERENCIA ENCONTRADA!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))
			except:
				if verbose:
					print('REGEX DETECTOU ERRONEAMENTE UM NÚMERO!\ntext:%s\ncurrent:%s\nfound:%s' % (self.texts[i], atuais[w], s))
					qtd_numero += 1

		w = 'parágrafo único'
		for s in matches(regexes[w], self.texts[i]):
			self.texts[i] = self.texts[i].replace(s, '[%s]' % w.upper(), 1)

	print('Erros: i:%d a:%d n:%d' % (qtd_inciso, qtd_alinea, qtd_numero))