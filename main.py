#!/usr/bin/env python
import persim
import pynmrstar
import ripser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from persim import wasserstein
from typing import List

def persistence_diagram(state: np.ndarray) -> np.ndarray:
    """
    Computes the persistence diagram of a single set of NMR chemical shifts.
    """
    return ripser.ripser(state, maxdim=1)['dgms']

def obtain_chemical_shifts(star: str) -> List[np.ndarray]:
    """
    Reads STAR NMR file format and parses out relevant information.
    """
    nmr_file = pynmrstar.Entry.from_file(star)
    
    states = []
    for chemical_shift in nmr_file.get_loops_by_category('Atom_chem_shift'):
        state = np.array(
                chemical_shift.get_tag(
                    ['Comp_index_ID', 'Comp_ID', 'Atom_ID', 
                        'Atom_type', 'Val', 'Val_err']
                    )
                )
        states.append(state)

    return states

def prep_shift_data(allshifts: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extracts raw shift data from list of experimental data and returns 
    list of persistence diagrams computed for each.
    """
    preprocessed = []
    for shift in allshifts:
        shift_data = shift[:,4].astype(np.float16)

        # ripser requires 2d arrays; for 1d data we must reshape
        if shift_data.ndim < 2:
            shift_data = np.reshape(shift_data, 
                                        (len(shift_data), 1))
        preprocessed.append(persistence_diagram(shift_data))

    return preprocessed

file1 = '2m6q_cs.str'
file2 = '7n82_cs.str'

shifts1 = obtain_chemical_shifts(file1)
shifts2 = obtain_chemical_shifts(file2)

states = [shifts1[0], shifts2[0]]

diagrams = prep_shift_data(states)

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
