import pynmrstar
import numpy as np
import matplotlib.pyplot as plt
import persim
import ripser
import seaborn as sns
from persim import wasserstein

# define function to generate persistence diagrams
def persistence_diagrams(states):
    diagrams = []
    for state in states:
        # compute persistence homology using ripser
        diagrams.append(ripser.ripser(state, maxdim=1)['dgms'])
    return diagrams

# read in NMR .str file using pyNMR-STAR
nmr_file = pynmrstar.Entry.from_file('path/to/nmr_file.str')

# extract the ensemble of states from the NMR file
states = []
for model in nmr_file.get_loops_by_category('_atom_site'):
    atoms = []
    for row in model:
        atoms.append([float(row['_atom_site.Cartn_x']),
                      float(row['_atom_site.Cartn_y']),
                      float(row['_atom_site.Cartn_z'])])
    states.append(np.array(atoms))

# generate persistence diagrams
diagrams = persistence_diagrams(states)

# compute pairwise Wasserstein distances between persistence diagrams
num_states = len(states)
similarity_matrix = np.zeros((num_states, num_states))
for i in range(num_states):
    for j in range(i+1, num_states):
        distance = wasserstein(diagrams[i], diagrams[j])
        similarity = np.exp(-distance**2)
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

# create a heatmap of the similarity matrix using seaborn
sns.set()
ax = sns.heatmap(similarity_matrix, cmap="YlGnBu")
plt.show()
