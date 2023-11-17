import sys
from ml.classifiers.dataloader import *
from scipy.stats import entropy
from scipy.sparse import vstack
from sklearn.cluster import AgglomerativeClustering, KMeans
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import minmax_scale

import numpy as np
from math import log

INF = 999999999

def calc_entropies(prob_lists):
    return [ entropy(probs) for probs in prob_lists ]


def calc_uncertainty(prob_lists):
    return [ 1 - max(probs) for probs in prob_lists ]


def info_amount(mat):
    non_zero_cols = len(mat.data)
    total_cols = mat.shape[1]
    return log(non_zero_cols + 1) / log(total_cols)



def calc_neighbor_dens(X, n_neighbors=100):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    neigh.fit(X)
    avg_sim = []
    for i in range(X.shape[0]):
        row = X[i,:]
        if type(X) == np.ndarray:
            row = row.reshape(1, -1)
        avg_sim.append(1 - np.mean(neigh.kneighbors(row)[0]))
    distances = neigh.kneighbors(X)[0]
    return np.array(avg_sim)



def calc_df(mat):
    df = {}
    coo = mat.tocoo()
    for doc, word in zip(coo.row, coo.col):
        if word not in df:
            df[word] = 0
        df[word] += 1
    return df


def info_amount2(mat, df, cache, ndocs):
    s = 0
    for word in mat.tocoo().col:
        if word not in df:
            continue
        f = df[word]
        if f in cache:
            val = cache[f]
        else:
            #frac = f / vocsize
            #val = - frac * log(frac)
            val = log(ndocs/f) 
            cache[f] = val
        s += val
    return s

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print ("usage: %s <Unlabeled set (U)> <output file> <alpha (ignored)> <budget> <ndim> <distance threshold> [#neighbors]" % sys.argv[0])
        sys.exit(-1)


unlab_filename = sys.argv[1]
#unlabprob_filename = sys.argv[2]

#lab_filename = sys.argv[3]
#labprob_filename = sys.argv[4]
outfilename = sys.argv[2]

alpha = float(sys.argv[3])
#beta = float(sys.argv[6])
budget = int(sys.argv[4])

ndim = int(sys.argv[5])

dist_thresh = float(sys.argv[6])

n_neighbors = 100
if len(sys.argv) == 8:
    n_neighbors = int(sys.argv[7])

print("unlabeled examples:", unlab_filename)

print("output:", outfilename)
print("budget:", budget)
#print("divers_factor:", divers_factor)
print("ndim:", ndim)
print("n_neighbors:", n_neighbors)

#Unlabeled data
X_U, y_U = load_sparse_data(unlab_filename, ndim=ndim)


#Labeled data
#X_L, y_L = load_sparse_data(lab_filename, ndim=ndim)
#probs_L = load_class_probabilities(labprob_filename)

X_L = csr_matrix( np.zeros( (1, ndim+1) ) )
lenU = X_U.shape[0]
lenL = 1


out = open(outfilename, "w", encoding="utf-8")
out_indices = open(outfilename + ".idx", "w", encoding="utf-8")


out_stats = open(outfilename + ".stats", "w")

print("U_size:", lenU)


#clusterer = AgglomerativeClustering(n_clusters=nselect, affinity="precomputed", linkage="average")
#clusterer = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=dist_thresh)


#print("Truncated SVD...")

#svd = TruncatedSVD(n_components=100)
#X_U_lowdim = svd.fit_transform(X_U)

#print("Computing Kernel Density Estimation...")

#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_U_lowdim)
#density = kde.score_samples(X_U_lowdim)
density = calc_neighbor_dens(X_U, n_neighbors=n_neighbors)
iamount = []
cache = {}
#density = minmax_scale(-density)

#df = calc_df(X_U)

for i in range(lenU):
    x = info_amount(X_U[i])
    #x = info_amount2(X_U[i], df, cache, lenU)
    iamount.append(x)
    #density[i] *= x

#density = np.ones(len(density)) - minmax_scale(density)
#density = minmax_scale(density)

for i in range(lenU):
    print(density[i], iamount[i], density[i]/iamount[i], file=out_stats)

#for i in range(lenU):
#    density[i] *= iamount[i]

out_stats.close()

selected = []
S = []

#score = alpha * non_redundancy + beta * uncertainty + gamma * info_density
scoreU = density 

while len(selected) < budget:

    ok = True
    s = np.argmax(scoreU)
    if scoreU[s] == -INF: #No more candidates
        break
    scoreU[s] = -INF #Grants that U[s] wont be selected again


    if len(S) > 0:
        dist_selected = cosine_distances(vstack(S), X_U[s])[:,0]
        if np.min(dist_selected) < dist_thresh:
            ok = False
    if ok:
        selected.append(s)
        S.append(X_U[s])


# Print selected
for i in selected:
    print(i, file=out_indices)
    print(y_U[i], file=out, end=" ")
    for j, value in zip(X_U[i].indices, X_U[i].data):
        print(j+1, end=":", file=out)
        print(value, end=" ", file=out)
    print(file=out)

out_indices.close()
out.close()


