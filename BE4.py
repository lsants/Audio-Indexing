### Auteurs : SANTIAGO Leonardo, MATTTE RIO FERNANDEZ Rodrigo

import numpy as np
import matplotlib.pyplot as plt

def enveloppe_energie(amplitude : list, taille_fen : int) -> list:
    offset = 0
    e = []
    while offset + taille_fen <= len(amplitude):
        f = amplitude[offset : offset + taille_fen]
        e.append(np.sqrt((f ** 2).sum() / taille_fen))
        offset += taille_fen // 2
    return np.array(e) 

def centroide_temporel(e : np.ndarray) -> np.ndarray:
    num = 0
    for t, el in enumerate(e):
        num += t * el
    tc = num / sum(e)
    return tc

def duree_effective(e : np.ndarray) -> int:
    seuil = 0.2 * e.max()
    ind = np.argwhere(e >= seuil)
    return len(ind)

def energie_globale(e : np.ndarray) -> float:
    eg = np.mean(e)
    return eg

def zcr(amplitude : list) -> int:
    amplitude1 = np.sign(amplitude[0 : -1])
    amplitude2 = np.sign(amplitude[1 : ])
    return (1/2) * np.sum(np.abs(amplitude2 - amplitude1)) 

def normalisation(x : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = x.mean(0)
    ec = x.std(0)
    x_norm = (x - m) / ec
    return x_norm, m, ec

def distance_descripteurs(titre_test : np.ndarray, vec_desc_app : np.ndarray, k : int) -> tuple[list,list]: 
    # entrées : - Titre à classifier (descripteur du signal de test);
    #           - Descripteurs normalisé de la matrice des données d'app.
    #           - k = nombre de voisins choisis
    # sorties : - vecteur avec 24 distances euclidiennes (titre - données d'app.);
    #           - vecteur avec les k titres les plus semblables.
    distances = np.linalg.norm(vec_desc_app - titre_test, axis=1)
    dist_k_part = (np.argpartition(distances,k))[:k] # partition pour trier les k distances le plus proches
    noms_app = np.asarray(np.genfromtxt("noms_app.txt", dtype=str, delimiter= '\n')) # Vecteur avec le nome de chaque titre d'apprentissage
    noms_hash = {indice : nom for indice,nom in zip(range(np.size(noms_app)),noms_app)} # Associe chaque indice à son titre
    k_titres_pp = np.asarray([noms_hash[indice] for indice in dist_k_part]) # Vecteur avec les k titres les plus proches
    return distances, k_titres_pp

def classification_kppv(titre_test : np.ndarray, vec_desc_app : np.ndarray, k : int) -> int: 
    # entrées : - Titre à classifier (descripteur du signal de test);
    #           - Descripteurs normalisé de la matrice des données d'app.
    #           - k = nombre de voisins choisis
    # sortie : Intier indiquant la classe du titre

    # Association de chaque titre d'apprentissage à sa classe (titre -> classe)
    noms_app = np.asarray(np.genfromtxt("noms_app.txt", dtype=str, delimiter= '\n'))    
    classes_app = np.loadtxt("classes_app.txt")
    classes_hash = {nom : int(classe) for nom,classe in zip(noms_app,classes_app)}

    # Calcul des k distances les plus proches
    k_titres_pp = distance_descripteurs(titre_test, vec_desc_app, k)[1] # On prend la deuxième valeur fournie par la fonction
    k_classes_pp = np.asarray([classes_hash[titre] for titre in k_titres_pp])
    # Determination de la classe avec le plus grand nombre d'examples
    classes_uniques, repetitions = np.unique(k_classes_pp, return_counts=True) # classes_uniques = nb. k de classes (uniques) trouvées ; repetitions = nb de repetitions de chaque element de "classes_uniques"
    indice_max_rep = np.argmax(repetitions) # Indice qui correspond à la classe la plus fréquente
    classification = classes_uniques[indice_max_rep] # Classe la plus fréquente

    return classification

def performance(classifications : np.ndarray) -> float:
    classes_test = np.loadtxt("classes_test.txt")
    class_correctes = np.where(classifications == classes_test)
    taux_reconnaissance = np.size(class_correctes) / np.size(classifications)
    return taux_reconnaissance

if __name__ == "__main__":
    donnees_app = np.loadtxt("donnees_app.txt")
    sa = donnees_app[1, :]
    donnees_test = np.loadtxt("donnees_test.txt")
    st = donnees_test[1, :]
    taille_fenetre = 300
    
    desc_app = np.zeros((donnees_app.shape[0], 4))
    desc_test = np.zeros((donnees_test.shape[0], 4))

    for i in range(donnees_app.shape[0]):
        sa = donnees_app[i,:]
        ea = enveloppe_energie(sa,taille_fenetre)
        a = centroide_temporel(ea)

        desc_app[i,0] = centroide_temporel(ea)
        desc_app[i,1] = duree_effective(ea)
        desc_app[i,2] = energie_globale(ea)
        desc_app[i,3] = zcr(sa)

    desc_app, m, ec = normalisation(desc_app)
    # np.savetxt("desc_app.txt", desc_app)
    # np.savetxt("normalisation_app.txt", (m, ec))

    for i in range(donnees_test.shape[0]):
        st = donnees_test[i, :]
        et = enveloppe_energie(st, taille_fenetre)

        desc_test[i,0] = centroide_temporel(et)
        desc_test[i,1] = duree_effective(et)
        desc_test[i,2] = energie_globale(et)
        desc_test[i,3] = zcr(st)

    desc_test, m, ec = normalisation(desc_test)

    # np.savetxt("desc_test.txt", desc_test)
    # np.savetxt("normalisation_test.txt", (m, ec))

    perfs = []
    k_values = range(1,24)
    for k in k_values: # Calcul pour k entre 1 et 23
        classifications = []
        for i in range(np.size(desc_test[:,0])):
            classifications.append(classification_kppv(desc_test[i,:], desc_app, k))
        classifications = np.asarray(classifications)
        perf = 100 * performance(classifications) # En pourcentage
        perfs.append(perf) # Vecteur avec les performances pour chaque valeur de k
    plt.plot(k_values, perfs, 'r')
    plt.ylabel('Performance [%]')
    plt.ylim([0,100])
    plt.xlabel('k')
    plt.grid()
    plt.show
    plt.savefig('k_perf')
