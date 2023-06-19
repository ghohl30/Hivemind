import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

import hive

def plot_hexagons(adjacency_matrix, mount_matrix, firstPiece, size=1):


    directions = {
        1: np.array([3 / 2 * size, -np.sqrt(3) / 2 * size]),
        2: np.array([0, -np.sqrt(3) * size]),
        3: np.array([-3 / 2 * size, -np.sqrt(3) / 2 * size]),
        4: np.array([-3 / 2 * size, np.sqrt(3) / 2 * size]),
        5: np.array([0, np.sqrt(3) * size]),
        6: np.array([3 / 2 * size, np.sqrt(3) / 2 * size])
    }

    labels = ['wQ1', 'wA1', 'wA2', 'wA3', 'wG1', 'wG2', 'wG3', 'wB1', 'wB2', 'wS1', 'wS2', 
              'bQ1', 'bA1', 'bA2', 'bA3', 'bG1', 'bG2', 'bG3', 'bB1', 'bB2', 'bS1', 'bS2']

    positions = {label: np.array([0, 0]) for label in labels} 

    fig, ax = plt.subplots(1)

    hexagons = {}
    stack = []
    placed = []

    if firstPiece[0,0] != 0:
        label = labels[firstPiece[0,0]]
    else:
        # get get the first label for placed pieces (idx is the row number of the first row that is not all zeros)
        # Get the indices of the non-zero elements
        nonzero_indices = np.nonzero(adjacency_matrix)

        # Get the first row index
        row_idx = nonzero_indices[0][0]

        label = labels[row_idx]
    

    positions[label] = np.array([0, 0])

    hexagons[label] = RegularPolygon(positions[label].tolist(), numVertices=6, radius=1, orientation=np.radians(30),
                        edgecolor='k', facecolor='none')
    
    ax.add_patch(hexagons[label])
    ax.text(positions[label][0], positions[label][1], label, ha='center', va='center')

    stack.append(label)
    placed.append(label)
    while len(stack) > 0:
        current = stack.pop()
        rowIdx = labels.index(current)
        for colIdx in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[rowIdx, colIdx] != 0 and labels[colIdx] not in placed:
                dir = directions[adjacency_matrix[rowIdx, colIdx]]
                positions[labels[colIdx]] = positions[labels[rowIdx]] + dir
                hexagons[labels[colIdx]] = RegularPolygon(positions[labels[colIdx]].tolist(), numVertices=6, radius=1, orientation=np.radians(30), 
                                     edgecolor='k', facecolor='none')
                ax.add_patch(hexagons[labels[colIdx]])
                ax.text(positions[labels[colIdx]][0], positions[labels[colIdx]][1], labels[colIdx], ha='center', va='center')
                stack.append(labels[colIdx])
                placed.append(labels[colIdx])

    # color the mount hexagons
    for i in range(mount_matrix.shape[0]):
        for j in range(mount_matrix.shape[1]):
            if mount_matrix[i,j] == 1 and np.sum(mount_matrix[i,:]) == 1:
                hexagons[labels[i]].set_facecolor('blue')

    # Calculate dynamic x and y limits
    all_positions = np.array(list(positions.values()))
    x_min, y_min = np.min(all_positions, axis=0) - size 
    x_max, y_max = np.max(all_positions, axis=0) + size

    ax.set_xlim([x_min, x_max]) # type: ignore
    ax.set_ylim([y_min, y_max]) # type: ignore
    ax.axis('off')


if __name__ == "__main__":
    # load adjacency matrix from csv
    A = np.loadtxt(open("adjacency_matrix.csv", "rb"), delimiter=",", skiprows=0)
    A[7,7] = -1
    A[8,8] = 1
    M = np.loadtxt(open("mount_matrix.csv", "rb"), delimiter=",", skiprows=0)
    firstPiece = np.loadtxt(open("firstPiece.csv", "rb"), delimiter=",", skiprows=0)
    plot_hexagons(A, M, firstPiece, size=1)