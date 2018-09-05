import numpy as np
import matplotlib.pyplot as plt
import pathlib
from dedalus.extras import plot_tools
import os
from dedalus.tools import post
import h5py

dir_path = os.path.dirname(os.path.realpath(__file__))
set_paths = list(pathlib.Path(dir_path+'/'+'snapshots').glob("snapshots_s*/snapshots_s*.h5"))
post.merge_sets(dir_path+'/'+'snapshots'+'/'+'snapshots.h5', set_paths, cleanup=True)

fig = plt.figure(figsize=(10, 6))
with h5py.File(dir_path+'/'+'snapshots/snapshots.h5', mode='r') as file:
  t = file['scales']['sim_time']
  print(file['scales'])

#xmesh, ymesh = plot_tools.quad_mesh(x=domain.grid(0, scales=domain.dealias).flatten(), y=domain.grid(2, scales=domain.dealias).flatten())
#plt.pcolormesh(xmesh, ymesh, b_array[80][:][1][:].T, cmap='RdBu_r')
#plt.colorbar()
#plt.xlabel('x')
#plt.ylabel('z')
#plt.show()


