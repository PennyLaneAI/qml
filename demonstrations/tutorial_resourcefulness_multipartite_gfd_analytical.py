import numpy as np
import math
import matplotlib.pyplot as plt

# Exact GFD purities of relevant states, as per Section IV.C in https://arxiv.org/pdf/2506.19696

## Multipartite Entanglement QRT
def binomial(n:int, k:int):
    """
    Extension of math comb
    """
    if k < 0:
        return 0
    else:
        return math.comb(n, k)

### free states
def me_free_purities(n: int):
    """
    binomial distribution
    """
    return np.array([binomial(n, k) for k in range(n+1)])/(2**n)

def me_ghz_purities(n: int):
    """
    binomial distribution over even weight modules,
    leftover purity gets pushed to largest module
    """
    pur = np.zeros(n+1)
    pur[range(0, n+1, 2)] = np.array([binomial(n, k) for k in range(0, n+1, 2)])/(2**n)
    pur[-1] += 0.5

    return pur

def me_w_purities(n: int):
    """
    weighted binomial distribution
    """
    return np.array([((n-2*k)**2 + 8*binomial(k,k-2)) * binomial(n, k) for k in range(n+1)])/((n**2) * (2**n))

def me_haar_purities(n: int):
    """
    weingarten distribution
    """
    pur = np.array([(3**k) * binomial(n, k) for k in range(n+1)])/((2**n) * (2**n + 1))
    pur[0] = 1 / (2**n)
    return pur

def me_ame_purities(n: int):
    """
    NB: AME STATES ARE KNOWN NOT TO EXIST FOR n=4 and n>6
    """
    pur = np.zeros(n+1)
    pur[0] = 1 / (2**n)
    pur[-1] = 1 - 1 / (2**n)
    return pur

def plot_purities_analytical(n: int):
    # generate plotting data
    data_series = [
        me_free_purities(n),
        me_ghz_purities(n),
        me_w_purities(n),
        me_haar_purities(n),
    ]
    # labels
    labels = [
        "Free",
        "GHZ",
        "W",
        "Haar"
    ]
    
    # Grab default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create two vertically aligned subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    for i, data in enumerate(data_series):
        color = colors[i % len(colors)]
        ax1.plot(data, label=f'{labels[i]}', color=color)
        ax2.plot(np.cumsum(data), label=f'{labels[i]}', color=color)
    
    ax1.set_ylabel('Purity')
    ax2.set_ylabel('Cumulative Purity')
    ax2.set_xlabel('Module weight')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Example

plot_purities_analytical(2)

