#!/usr/bin/env python
# coding: utf-8

# # Ce programme détecte les maxima d'un signal sinusoidal décroissant et exporte un fichier [fichier]_maxima_.csv
# # contenant l'amplitude des maxima et l'écart entre chaque maximum (demi-période x 2)
# 
# **Attention :** pour un résultat joli : prendre une résolution d'environ 10 ms !


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### Entrer le nom du fichier
file = "Pendule.csv"

data = pd.read_csv(file,sep=";",decimal=",")

data = data.to_numpy().transpose()
print('resolution',np.mean(np.diff(data[0]))*1000,'ms')

### Limiter éventuellement la plage d'acquisition :

#data = data[:,:10000]

### Refaire évenutellement le zero

#data[1] = data[1] - np.mean(data[1])




### Détection des zeros et interpolation de leur valeurs

zero_list = np.where((np.diff(np.sign(data[1]))))[0]
zero_list_precision = data[0][zero_list] - data[1][zero_list]*(data[0][zero_list+1]-data[0][zero_list])/(data[1][zero_list+1]-data[1][zero_list])

### Recherche des maxima entre chaque zero

max_list = []
for i in range(len(zero_list)-1):
    max_list += [zero_list[i] + np.argmax((data[1]**2)[zero_list[i]:zero_list[i+1]]),]
max_list = np.array(max_list, dtype=int)

### Controle qualité

plt.plot(data[0],data[1])
plt.plot(zero_list_precision, 0*zero_list_precision, '.')
plt.plot(data[0][max_list],  data[1][max_list], '.')
plt.xlim(10,20)

plt.show()

### Calcul de la demi-période entre deux maxima

period_list = 2*np.abs(zero_list_precision[1:] - zero_list_precision[:-1])
amplitude_list = np.abs(data[1][max_list])

#L'erreur est déterminée comme l'écart entre deux 

### Si il y a des points parasites à des max faibles, un seuil les enlève

threshold = 2.  # seuil en °
period_list = period_list[amplitude_list > threshold]
max_list = max_list[amplitude_list > threshold]
amplitude_list = amplitude_list[amplitude_list > threshold]

### Export des données en CSV (séparateur tabulation)
pd.DataFrame(data = np.transpose([data[0][max_list], amplitude_list, period_list]),
             columns=['Temps','Amplitude','Periode']).to_csv(
             file[:-4] + '_maxima_' + file[-4:],
             sep='\t',decimal=".", index=False)

### Affichage des données

# Les amplitudes
plt.figure()
plt.plot(data[0][max_list],amplitude_list,'.')
plt.yscale('log')
plt.ylim(1,90)

plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (°)')

# Les périodes
plt.figure()
plt.plot(amplitude_list**2,period_list,'.')
plt.xlabel('Amplitude² (°²)')
plt.ylabel('Periode (s)')

plt.show()