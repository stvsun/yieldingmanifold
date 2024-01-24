import numpy as np
import argparse
import gmpy2
import os

parser = argparse.ArgumentParser()
parser.add_argument('case_id', type=int)
args = parser.parse_args()
case_id = args.case_id
# file names
dir_name = f'yield-surface-moredense-{case_id}/'
inp_prefix = dir_name + f'yld-surf-d-{case_id}'
out_prefix = dir_name + 'recon_point_cloud'

indices = np.loadtxt(inp_prefix + '_indices.txt')
vertices = np.loadtxt(inp_prefix + '_vertices.txt')
normals = np.loadtxt(inp_prefix + '_normals.txt')

# surface scales
scale_dict = {
  0   : np.array([20.6922493, 19.84395027, 18.52919006*np.sqrt(2)]),
  1100: np.array([24.05110931, 24.7630806, 22.44108963*np.sqrt(2)]),
  1110: np.array([23.27001953, 22.43214035, 21.24725914*np.sqrt(2)]),
  2100: np.array([28.18310928, 29.00629997, 27.18322945*np.sqrt(2)]),
  2110: np.array([26.17239952, 25.40192032, 23.67305946*np.sqrt(2)])
}

scale = scale_dict[case_id]
vertices *= scale
normals /= scale
for i in range(normals.shape[0]):
    n_ = np.linalg.norm(normals[i,:])
    normals[i,:] /= n_
    if np.dot(normals[i,:], vertices[i,:]) < 0:
        normals[i,:] *= -1

num_patch = int(indices.max())

n_points = vertices.shape[0]
f = open(out_prefix+'.vtk', 'wt')
f.write('# vtk DataFile Version 2.0\nGrain example\nASCII\nDATASET UNSTRUCTURED_GRID\n')
f.write('POINTS {} float\n'.format(n_points))
for i in range(n_points):
    f.write('{:.8g} {:.8g} {:.8g}\n'.format(vertices[i,0], vertices[i,1], vertices[i,2]))

f.write('\nCELLS {} {}\n'.format(n_points, n_points * 2))
for i in range(n_points):
    f.write('{} {}\n'.format(1, i))
f.write('\nCELL_TYPES {}\n'.format(n_points))
for i in range(n_points):
    f.write('1\n')

f.write('\nPOINT_DATA {}\n'.format(n_points))
f.write('VECTORS normals float\n')
for i in range(n_points):
    f.write('{:.8g} {:.8g} {:.8g}\n'.format(normals[i,0],normals[i,1],normals[i,2]))
f.close()

out_dir = dir_name + 'datafiles'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for p in range(num_patch+1):
    v = vertices[abs(indices - p) < 1e-8, :]
    n = normals[abs(indices - p) < 1e-8, :]
    
    n_points = v.shape[0]
    Ngrid = int(gmpy2.isqrt(n_points))
    n_cells = int((Ngrid-1) * (Ngrid-1))
    f = open(out_dir+'/patch_{}.vtk'.format(p), 'wt')
    f.write('# vtk DataFile Version 2.0\nGrain example\nASCII\nDATASET UNSTRUCTURED_GRID\n')
    f.write('POINTS {} float\n'.format(n_points))
    for i in range(n_points):
        f.write('{:.8g} {:.8g} {:.8g}\n'.format(v[i,0], v[i,1], v[i,2]))

    f.write('\nCELLS {} {}\n'.format(n_cells, n_cells * 5))
    for i,j in np.ndindex(Ngrid-1, Ngrid-1):
        orig_id = j * Ngrid + i
        f.write('4 {} {} {} {}\n'.format(orig_id, orig_id+1, orig_id+Ngrid+1, orig_id+Ngrid))
    f.write('\nCELL_TYPES {}\n'.format(n_cells))
    for i in range(n_cells):
        f.write('9\n')

    f.write('\nPOINT_DATA {}\n'.format(n_points))
    f.write('VECTORS normals float\n')
    for i in range(n_points):
        f.write('{:.8g} {:.8g} {:.8g}\n'.format(n[i,0],n[i,1],n[i,2]))
    f.close()
