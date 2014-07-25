from mdtraj.utils import timing
import numpy as np
from mixtape.datasets._muller import propagate, muller_potential
import matplotlib.pyplot as pp


def plot_v(minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, **kwargs):
    "Plot the Muller potential"
    grid_width = max(maxx-minx, maxy-miny) / 200.0
    
    ax = kwargs.pop('ax', None)
    
    xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
    V = muller_potential(xx, yy)
    # clip off any values greater than 200, since they mess up
    # the color scheme
    if ax is None:
        ax = pp

    ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)

with timing():
    trj = propagate(n_steps=1000000, thin=100, kT=1.5e4, D=0.010, dt=0.1)
print(trj.shape)
plot_v()
pp.plot(trj[:, 0], trj[:, 1], marker='.', c='k')
pp.show()
