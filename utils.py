import matplotlib.pyplot as plt
import numpy as np



def plot_octree(node, ax):
    if node.children is None:
        lower = node.center - node.size / 2
        rect = plt.Rectangle(lower, node.size, node.size, fill=False, edgecolor='black')
        ax.add_patch(rect)
    else:
        for child in node.children:
            plot_octree(child, ax)


