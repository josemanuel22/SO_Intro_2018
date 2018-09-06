"""
Plot planes from joint analysis files.

Usage:
    rb_convection_plot.py <file> [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from dedalus.extras import plot_tools
import os
from dedalus.tools import post
import h5py
import dedalus.public as de

Nx=16; Ny= 16; Nz = 16;
Lx=25; Ly=25; Lz= 1;
x_basis = de.Fourier('x', Nx, interval=(0, Lx))
y_basis = de.Fourier('y', Ny, interval=(0, Ly))
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz))
domain = de.Domain([x_basis,y_basis,z_basis], grid_dtype=np.float64)

dir_path = os.path.dirname(os.path.realpath(__file__))
set_paths = list(pathlib.Path(dir_path+'/'+'snapshots').glob("snapshots_s*/snapshots_s*.h5"))
if not os.path.isfile(dir_path+'/'+'snapshots'+'/'+'snapshots.h5'):
  post.merge_sets(dir_path+'/'+'snapshots'+'/'+'snapshots.h5', set_paths, cleanup=True)

def interpolate(filename, output, time, axis, value):
  fig = plt.figure(figsize=(10, 6))
  with h5py.File(dir_path+'/'+'snapshots/snapshots.h5', mode='r') as file:
    t = file['scales']['sim_time']
    b = file['tasks']['b']
    if axis =='x':
      b_v=b[time,value,:,:].T
    elif axis =='y':
      b_v=b[time,:,value,:].T
    elif axis =='z':
      b_v=b[time,:,:,value].T
    else:
      return
    xmesh, ymesh = plot_tools.quad_mesh(x=domain.grid(0, scales=domain.dealias).flatten(), y=domain.grid(2, scales=domain.dealias).flatten())
    plt.pcolormesh(xmesh, ymesh, b_v, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')

    dpi=100
    print(filename)
    print(output)
    savepath = output.joinpath(filename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
  from docopt import docopt

  args = docopt(__doc__)
  output_path = pathlib.Path(args['--output']).absolute()
  if not output_path.exists():
    output_path.mkdir()
    interpolate(args['<file>'], output_path, time=80, axis='y', value=8)

  #plt.show()

