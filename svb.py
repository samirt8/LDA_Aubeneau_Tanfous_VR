
# SVB

# TODO :
# 		 - utiliser le corpus de textes utilisé par les chercheurs
#		 - utiliser les meme variables que vous
#		 - pour l'instant, j'ai fait l'hypothese que tous les documents ont la meme taille
#		 - eventuellement, ajouter un critere de convergence ???
#
# je fait les modifs ce soir.
#

K = 6
alpha_const = 0.5
beta_const = 1.
nb_iter = 100

import pandas as pd
import numpy as np
import math
import scipy.special as ss
import random

#
# creation d'un mini-corpus de documents
#

documents_list = []

documents_list.append( pd.Series(["antilope","mouton","chat","chien","cheval","chat"]))
documents_list.append( pd.Series(["antilope","lion","girafe","chien","cheval","chat"]))
documents_list.append( pd.Series(["boeuf","lion","coq","chien","cochon","chat"]))
documents_list.append( pd.Series(["antilope","boeuf","chien","chien","cheval","mouton"]))

documents_list.append( pd.Series(["main","tete","oreille","jambe","bras","ventre"]))
documents_list.append( pd.Series(["main","tete","oreille","tete","doigt","bras"]))

documents_list.append( pd.Series(["voiture","camion","bateau","avion","helicoptere","moto"]))
documents_list.append( pd.Series(["voiture","camion","bateau","voiture","velo","moto"]))
documents_list.append( pd.Series(["marche","moto","helicoptere","voiture","velo","voiture"]))

documents_list.append( pd.Series(["france","angleterre","italie","egypte","mali","nigeria"]))
documents_list.append( pd.Series(["russie","france","italie","egypte","usa","france"]))
documents_list.append( pd.Series(["russie","angleterre","italie","egypte","usa","nigeria"]))
documents_list.append( pd.Series(["russie","usa","espagne","france","usa","nigeria"]))

#documents_list.append( pd.Series(["france","chef","cuisine","mouton","boeuf","sauce"]))
#documents_list.append( pd.Series(["sauce","chef","cuisine","assiette","poivre","sauce"]))
#documents_list.append( pd.Series(["sauce","france","boeuf","assiette","poivre","fromage"]))
#documents_list.append( pd.Series(["vin","sauce","coq","assiette","sel","poivre"]))

#documents_list.append( pd.Series(["but","gazon","tir","football","gardien","ballon"]))
#documents_list.append( pd.Series(["but","tir","football","football","attaquant","ballon"]))

#documents_list.append( pd.Series(["antilope","lion","girafe","egypte","mali","afrique"]))
#documents_list.append( pd.Series(["tir","france","italie","football","angleterre","ballon"]))

documents_list.append( pd.Series(["football","tennis","equitation","basket","aviron","boxe"]))
documents_list.append( pd.Series(["football","tennis","football","velo","pingpong","course"]))
documents_list.append( pd.Series(["basket","aviron","equitation","boxe","boxe","course"]))

documents_list.append( pd.Series(["crayon","feuille","stylo","imprimante","gomme","crayon"]))
documents_list.append( pd.Series(["trousse","scotch","stylo","crayon","gomme","feuille"]))

documents_df = pd.DataFrame(documents_list).transpose()
nb_words_by_document = documents_df.shape[0]
nb_documents = documents_df.shape[1]

#
# creation de la liste de mots
#

words_set = set()
for i in range(0, nb_documents ):
    for j in range(0, nb_words_by_document):
        words_set.add(documents_df.iloc[j,i])
words_list = list(words_set)
words_df = pd.Series(words_list)

nb_words = len(words_set)

mot = pd.DataFrame(0,  
                      index=np.arange(nb_words_by_document),
                      columns=np.arange(nb_documents))
for d in range(0,nb_documents):
    for n in range(0,nb_words_by_document):
        mot.at[n, d] = words_list.index(documents_df.at[n, d])
        
#
# calcul des frequences des mots
# (non-utlise dans cette version de l'algo)
#

frequence = pd.DataFrame(0.,  
                      index=np.arange(nb_words),
                      columns=np.arange(nb_documents))
for d in range(0,nb_documents):
    for j in range(0,nb_words):
        somme = 0
        for n in range(0,nb_words_by_document):
            if( documents_df.iloc[n,d] == words_df[j]):
                somme+=1
        frequence.at[j, d] = somme / nb_words_by_document

		
#
# initialisations des matrices alpha, beta et gamma
#

gamma = pd.DataFrame(0.,  
                      index=np.arange(nb_words_by_document*K),
                      columns=np.arange(nb_documents))
for d in range(0,nb_documents):
    for s in range(0,nb_words_by_document*K):
        gamma.at[s, d] = 1/K + random.gauss(0,1)

alpha = pd.DataFrame(0.,  
                      index=np.arange(K),
                      columns=np.arange(nb_documents))
for d in range(0,nb_documents):
    for i in range(0,K):
        alpha.at[i, d] = alpha_const

beta = pd.DataFrame(0.,  
                      index=np.arange(K),
                      columns=np.arange(nb_words))
for i in range(0,K):
    for j in range(0,nb_words):
        beta.at[i, j] = beta_const

#
# corps de l'algo svb
#

for c in range(0,nb_iter):
    for j in range(0,nb_documents):
        for k in range(0,K):
            somme = 0
            for i in range(0,nb_words_by_document):
                s=i*K+k
                somme += gamma.iloc[s,j]
            alpha.at[k,j]= alpha_const + somme
    for k in range(0,K):
        for w in range(0,nb_words):
            somme = 0
            for i in range(0,nb_words_by_document):
                for j in range(0,nb_documents):
                    if( documents_df.iloc[i,j]==words_df[w]):
                        s=i*K+k
                        somme += gamma.iloc[s,j]
            beta.at[k,w]= beta_const + somme
    for i in range(0,nb_words_by_document):
        for j in range(0,nb_documents):
            somme_gamma = 0
            for k in range(0,K):
                t1 = ss.digamma(alpha.iloc[k,j])
                w = mot.iloc[i,j]
                t2 = ss.digamma(beta.iloc[k,w])
                somme = 0
                for w in range(0,nb_words):
                    somme += beta.iloc[k,w]
                t3 = ss.digamma(somme)
                s=i*K+k
                gamma.at[s,j]=math.exp(t1+t2-t3)
                somme_gamma += gamma.at[s,j]
            for k in range(0,K):
                s=i*K+k
                gamma.at[s,j]=gamma.iloc[s,j]/somme_gamma

#
# calcul des topics pour visualiser l'efficacite de l'algo
#

phi = pd.DataFrame(0.,  
                      index=np.arange(K),
                      columns=np.arange(nb_words))
for w in range(0,nb_words):
    dirichletParam = beta[w]
    distr = np.random.dirichlet( dirichletParam )
    phi[w] = distr
    
topic_list = []
for k in range(0,K):
    topic = []
    for w in range(0,nb_words):
        if( phi.iloc[k,w] == max(phi[w])):
            topic.append(words_df[w])
    topic_list.append(topic)