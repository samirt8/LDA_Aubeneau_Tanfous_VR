
import pandas as pd
import numpy as np
import math
import scipy.special as ss
import random
import time



docword_KOS = pd.read_csv('docword.kos.txt', sep=" ")
vocab_KOS= pd.read_csv('vocab.kos.txt', sep=" ")

# corpus doit etre un data frame de 3 colonnes avec:
#  colonne 1 : id du document
#  colonne 2 : id du mot
#  colonne 3 : nbr d'occurence du mot dans le document

# vocabulaire doit etre un dataframe avec 1 seule colonne:
#  tq 1+indice(mot)=id du mot
def svb( corpus, vocabulaire, nbr_classes, nbr_iterations, alpha, beta)

	K = nbr_classes
	alpha_const = alpha
	beta_const = beta
	nb_iter = nbr_iterations
	
	nb_lines = corpus.shape[0]
	nb_documents = corpus.iloc[nb_lines-1,0]
	nb_words = vocabulaire.shape[0]

	documents_list = []
	line = 0
	nb_words_by_document = pd.Series(0, index=np.arange(nb_documents))
	documents_df = pd.DataFrame(index=np.arange(1000), columns=np.arange(nb_documents))
	index_words = dict((k,[]) for k in np.arange(nb_words))
	for i in range(0,nb_lines):
		no_document = corpus.iloc[i,0]-1
		for j in range(0,corpus.iloc[i,2]):
			no_word = corpus.iloc[i,1]-1
			documents_df.at[nb_words_by_document[no_document],no_document] = no_word
			list_index = index_words.get(no_word)
			list_index.append((no_document,nb_words_by_document[no_document]))
			index_words[no_word] = list_index
			nb_words_by_document[no_document] += 1
	#	if( i%10000 == 0 ):
	#		print(i)

	max_words_by_document = max(nb_words_by_document)

	gamma = pd.DataFrame(0.,  
						  index=np.arange(max_words_by_document*K),
						  columns=np.arange(nb_documents))
	for d in range(0,nb_documents):
		for s in range(0,nb_words_by_document[d]*K):
			gamma.at[s, d] = 1/K + random.gauss(0,1)

	alpha = pd.DataFrame(0.,  
						  index=np.arange(K),
						  columns=np.arange(nb_documents))
	for d in range(0,nb_documents):
		for i in range(0,K):
			alpha.at[i, d] = alpha_const

	beta = pd.DataFrame(0.,
						  index=np.arange(nb_words),  
						  columns=np.arange(K))
	for i in range(0,K):
		for j in range(0,nb_words):
			beta.at[j, i] = beta_const

	# corps de l'algo svb
	#

	for c in range(0,nb_iter):
	#	print( c )
	#	print( " boucle 1 ")
		for j in range(0,nb_documents):
			for k in range(0,K):
				somme = 0
				for i in range(0,nb_words_by_document[j]):
					s=i*K+k
					somme += gamma.iloc[s,j]
				alpha.at[k,j]= alpha_const + max( 0., somme)
	#	print( " boucle 2 ")
		for k in range(0,K):
			for w in range(0,nb_words):
				somme = 0
				for t in index_words.get(w):
					j = t[0]
					i = t[1]
					s=i*K+k
					somme += gamma.iloc[s,j]
	#            for j in range(0,nb_documents):
	#                for i in range(0,nb_words_by_document[j]):
	#                    if( documents_df.iloc[i,j]==w):
	#                        s=i*K+k
	#                        somme += gamma.iloc[s,j]
				beta.at[w,k]= beta_const + max( 0., somme)
	#	print( " boucle 3 ")
		for j in range(0,nb_documents):
			for i in range(0,nb_words_by_document[j]):
				somme_gamma = 0
				for k in range(0,K):
					t1 = ss.digamma(alpha.iloc[k,j])
					w = documents_df.iloc[i,j]
					t2 = ss.digamma(beta.iloc[w,k])
					somme = beta[k].sum()
	#                for w in range(0,nb_words):
	#                    somme += beta.iloc[k,w]
					t3 = ss.digamma(somme)
					s=i*K+k
					gamma.at[s,j]=math.exp(t1+t2-t3)
					somme_gamma += gamma.at[s,j]
				for k in range(0,K):
					s=i*K+k
					gamma.at[s,j]=gamma.iloc[s,j]/somme_gamma

	return ( alpha, beta, gamma)

triplet = svb( docword_KOS, vocab_KOS, 10, 10, 0.5, 0.5)
#
# calcul des topics pour visualiser l'efficacite de l'algo
#
beta = triplet[1]
beta2 = beta.transpose()
phi = pd.DataFrame(0.,  
					  index=np.arange(K),
					  columns=np.arange(nb_words))
for w in range(0,nb_words):
	dirichletParam = beta2[w]
	distr = np.random.dirichlet( dirichletParam )
	phi[w] = distr
	
topic_list = []
for k in range(0,K):
	topic = []
	for w in range(0,nb_words):
		if( phi.iloc[k,w] == max(phi[w])):
			topic.append(vocabulaire.iloc[w,0])
	topic_list.append(topic)
