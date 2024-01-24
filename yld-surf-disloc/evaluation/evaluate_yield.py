import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
import argparse

# surface scales
scale_dict = {
  0   : np.array([20.6922493, 19.84395027, 18.52919006*np.sqrt(2)]),
  1100: np.array([24.05110931, 24.7630806, 22.44108963*np.sqrt(2)]),
  1110: np.array([23.27001953, 22.43214035, 21.24725914*np.sqrt(2)]),
  2100: np.array([28.18310928, 29.00629997, 27.18322945*np.sqrt(2)]),
  2110: np.array([26.17239952, 25.40192032, 23.67305946*np.sqrt(2)])
}


class MLP(nn.Module):
  def __init__(self, in_dim: int, out_dim: int):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.fc1 = nn.Linear(in_dim, 128)
    self.fc2 = nn.Linear(128, 256)
    self.fc3 = nn.Linear(256, 128)
    self.out = nn.Linear(128, out_dim)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.out(x)
    return x


class Compute_project:
  def __init__(self, patch_models, patch_tx, surface_scale):
    self.phi = patch_models
    self.transform = patch_tx
    self.surface_scale = surface_scale
  def config(self, target, patch_id):
    self.target = np.copy(target)
    self.idx_i = np.copy(patch_id)
    self.phi_i = self.phi[patch_id]
    translate_i, scale_i, rotate_i = self.transform[patch_id]
    self.trans_i = translate_i
    self.scale_i = scale_i
    self.rotat_i = rotate_i
  def compute(self):
    def phi(v):
      y = self.phi_i( torch.tensor(v.reshape(1,-1)).float() )
      yt = (y.squeeze() @ self.rotat_i.transpose(0, 1)) / self.scale_i - self.trans_i
      return yt.detach().numpy().reshape(-1) * self.surface_scale
    def phi1(v):
      return phi(v)[0]
    def phi2(v):
      return phi(v)[1]
    def phi3(v):
      return phi(v)[2]
    def phi_grad(v):
      phi1_p = opt.approx_fprime(v, phi1, epsilon=1e-6)
      phi2_p = opt.approx_fprime(v, phi2, epsilon=1e-6)
      phi3_p = opt.approx_fprime(v, phi3, epsilon=1e-6)
      return np.stack([phi1_p, phi2_p, phi3_p])
    # projection objective function (Eucledian distance)
    def dist(v):
      diff = phi(v) - self.target
      #df_dv= phi_grad(v); print(phi(v) @ df_dv)
      return diff
    # optimization constraint
    x0 = np.array([0.6,0.6])
    res = opt.least_squares(dist, x0, jac=phi_grad, bounds=(0.,1.))
    soln = res.x
    proj = phi(soln)
    #print(res)
    #print('find projection point {}, with the normal vector {}'.format(proj, norm))
    return soln, proj


def bisect_search_residual(compt_proj, tar, idx):
  compt_proj.config(tar, idx)
  _, tar_proj = compt_proj.compute()
  sgn = -1. if np.linalg.norm(tar) < np.linalg.norm(tar_proj) else 1.
  return sgn*np.linalg.norm(tar-tar_proj)

def find_loc_on_surface(compt_proj, tar, idx):
    # implement a bisection search algo to find the actual crossing
    # initial guess range 15/45
    lb, ub = 10., 50.
    res_lb = bisect_search_residual(compt_proj, lb*tar, idx)
    res_ub = bisect_search_residual(compt_proj, ub*tar, idx)
    while True:
        mid = lb + (ub - lb) * abs(res_lb) / (abs(res_lb) + abs(res_ub))
        res_mb = bisect_search_residual(compt_proj, mid*tar, idx)
        if res_mb * res_lb > 0.:
            lb, ub = np.copy(mid), np.copy(ub)
            res_lb, res_ub = np.copy(res_mb), np.copy(res_ub)
        else:
            lb, ub = np.copy(lb), np.copy(mid)
            res_lb, res_ub = np.copy(res_lb), np.copy(res_mb)
        if (ub-lb) < 1e-7 or abs(res_mb) < 1e-5: break
    #compt_proj.config(l*tar, idx)
    _, tar_pred = compt_proj.compute()
    return tar_pred


# interpolate points given direction
def get_yield_results(gp_model_id, dir_fname):
  # find the number of patches
  gp_model_prefix = 'saved_models/yld-surf-d-{:d}'.format(gp_model_id)
  # load model
  phi_dict = torch.load(gp_model_prefix+'.pt')
  num_patches = len(phi_dict["patch_uvs"])
  phi = nn.ModuleList([MLP(2, 3) for i in range(num_patches)])
  phi.load_state_dict(phi_dict["final_model"])
  # get other patch informations
  patch_tx = phi_dict["patch_txs"]
  ctr_v = phi_dict["patch_ctr"]
  surface_scale = scale_dict[gp_model_id]
  
  # load stresses
  pres_sigs = np.loadtxt(dir_fname)
  pres_sigs[:,2] *= np.sqrt(2)
  # get stress directions
  pres_mags = np.linalg.norm(pres_sigs, axis=1)
  # penalize the stress magnitude greater than zero
  pres_mags = np.maximum(pres_mags, 1e-6* np.ones_like(pres_mags))
  pres_locs = (pres_sigs.T / pres_mags).T
  Ndir = pres_locs.shape[0]
  #Ndir = 1
  gp_points_result = np.zeros((Ndir,3))
  compt_proj = Compute_project(phi, patch_tx, surface_scale)
  for i in range(Ndir):
    tar_loc = pres_locs[i,:]
    ctr_dist = np.arccos( (np.dot(ctr_v,tar_loc)) / np.linalg.norm(ctr_v,axis=1) )
    idxes = np.argsort(ctr_dist)
    temp_points = [find_loc_on_surface(compt_proj, tar_loc, idxes[k]) for k in range(3)]
    #temp_points_norm = [np.dot(tar_loc, p) / np.linalg.norm(p) for p in temp_points]
    #gp_points_result[i,:] = temp_points[np.argmax(temp_points_norm)]
    gp_points_result[i,:] = (temp_points[0]+temp_points[1]+temp_points[2]) / 3.
    #print('find solution for projection problem:\npred = {}\ntar dir = {}'.format(gp_points_result[i,:], tar_loc))
  
  # compare stress norms to determine if a stress state is yielding or not
  gp_mags = np.linalg.norm(gp_points_result, axis=1)
  yield_result = (pres_mags >= gp_mags)
  
  return yield_result.astype(int)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("inp_fname")
parser.add_argument("out_fname")
parser.add_argument("case_id", type=int)
args = parser.parse_args()
case_id = args.case_id
inp_fname = args.inp_fname
out_fname = args.out_fname

# main stream
yield_result = get_yield_results(case_id, inp_fname)
np.savetxt(out_fname, yield_result[:, np.newaxis], fmt="%d")













