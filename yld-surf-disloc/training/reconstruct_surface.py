import argparse
import copy
import time

import numpy as np
import ot
import point_cloud_utils as pcu
import torch
import torch.nn as nn
from torch.autograd import grad
from fml.nn import SinkhornLoss, pairwise_distances
from scipy.spatial import cKDTree
from scipy.linalg import null_space
import scipy.optimize as opt

import utils


class MLP(nn.Module):
    """
    A simple fully connected network mapping vectors in dimension in_dim to vectors in dimension out_dim
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        #self.fc4 = nn.Linear(512, 512)
        self.out = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        #x = self.relu(self.fc4(x))
        x = self.out(x)
        return x

def prune_point_cloud_poisson_disk(x, n, r):
    idx = pcu.downsample_point_cloud_poisson_disk(x, r)
    return x[idx], n[idx]

def compute_patches(x, n, r, c, angle_thresh=91.0,  min_pts_per_patch=10, devices=('cpu',)):
    """
    Given an input point cloud, X, compute a set of patches (subsets of X) and parametric samples for those patches.
    Each patch is a cluster of points which lie in a ball of radius c * r and share a similar normal.
    The spacing between patches is roughly the radius, r. This function also returns a set of 2D parametric samples
    for each patch. These samples are used to fit a function from the samples to R^3 which agrees with the patch.

    :param x: A 3D point cloud with |x| points specified as an array of shape (|x|, 3) (each row is a point)
    :param n: Unit normals for the point cloud, x, of shape (|x|, 3) (each row is a unit normal)
    :param r: The approximate separation between patches
    :param c: Each patch will fit inside a ball of radius c * r
    :param angle_thresh: If the normal of a point in a patch differs by greater than angle_thresh degrees from the
                        normal of the point at the center of the patch, it is discarded.
    :param min_pts_per_patch: The minimum number of points allowed in a patch
    :param devices: A list of devices on which to store each patch. Patch i is stored on devices[i % len(devices)].
    :return: Two lists, idx and uv, of torch tensors, where uv[i] are the parametric samples (shape = (np, 2)) for
             the i^th patch, and idx[i] are the indexes into x of the points for the i^th patch. i.e. x[idx[i]] are the
             3D points of the i^th patch.
    """

    covered = np.zeros(x.shape[0], dtype=bool)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # prune_point_cloud_disk deprecated
    #ctr_v, ctr_n = pcu.prune_point_cloud_poisson_disk(x, n, r, best_choice_sampling=True)
    ctr_v, ctr_n = prune_point_cloud_poisson_disk(x, n, r)

    if len(ctr_v.shape) == 1:
        ctr_v = ctr_v.reshape([1, *ctr_v.shape])
        ctr_n = ctr_n.reshape([1, *ctr_n.shape])
    kdtree = cKDTree(x)
    ball_radius = c * r
    angle_thresh = np.cos(np.deg2rad(angle_thresh))
    # query patch neighborhood
    kdtree_c = cKDTree(ctr_v)
    idx_qc = kdtree_c.query(ctr_v[1,:], k=5)
    print('neighbors of patch {}: {}'.format(kdtree_c, idx_qc))

    patch_indexes = []
    patch_uvs = []
    patch_xs = []
    patch_transformations = []
    patch_centers = []

    def make_patch(v_ctr, n_ctr):
        idx_i = np.array(kdtree.query_ball_point(v_ctr, ball_radius, p=np.inf))
        #good_normals = np.squeeze(n[idx_i] @ n_ctr.reshape([3, 1]) > angle_thresh)
        #idx_i = idx_i[good_normals]

        if len(idx_i) < min_pts_per_patch:
            print("Rejecting small patch with %d points" % len(idx_i))
            return

        covered_indices = idx_i[np.linalg.norm(x[idx_i] - v_ctr, axis=1) < r]
        covered[covered_indices] = True

        uv_i = pcu.lloyd_2d(len(idx_i)).astype(np.float32)
        x_i = x[idx_i].astype(np.float32)
        translate_i = -np.mean(x_i, axis=0)
        patch_centers.append( -translate_i )
        #print('Sample center of patch {}: {}'.format(id_, patch_centers[id_,:]))

        device = devices[len(patch_xs) % len(devices)]
                
        scale_i = np.array([1.0 / np.max(np.linalg.norm(x_i + translate_i, axis=1))], dtype=np.float32)
        rotate_i, _, _ = np.linalg.svd((x_i + translate_i).T, full_matrices=False)
        transform_i = (torch.from_numpy(translate_i).to(device),
                       torch.from_numpy(scale_i).to(device),
                       torch.from_numpy(rotate_i).to(device))

        x_i = torch.from_numpy((scale_i * (x_i.astype(np.float32) + translate_i)) @ rotate_i).to(device)

        patch_transformations.append(transform_i)
        patch_indexes.append(torch.from_numpy(idx_i))
        patch_uvs.append(torch.tensor(uv_i, device=device, requires_grad=True))
        patch_xs.append(x_i)
        #print("Computed patch with %d points" % x_i.shape[0])
        
    for i in range(ctr_v.shape[0]):
        make_patch(ctr_v[i], ctr_n[i])

    for i in range(x.shape[0]):
        if np.sum(covered) == x.shape[0]:
            break
        if not covered[i]:
            make_patch(x[i], n[i])

    # assert np.sum(covered) == x.shape[0], "There should always be one at least one patch per input vertex"

    print("Found %d neighborhoods" % len(patch_indexes))
    patch_centers = np.array(patch_centers)
    return patch_indexes, patch_uvs, patch_xs, patch_transformations, patch_centers #, ctr_v


def patch_means(patch_pis, patch_uvs, patch_idx, patch_tx, phi, x, devices, num_batches):
    """
    Given a set of charts and pointwise correspondences between charts, compute the mean of the overlapping points in
    each chart. This is used to denoise the Atlas after each chart has beeen individually fitted.
    The charts may not agree exactly on their prediction, so we compute the mean predictions of overlapping charts
    and fit each chart to that mean.

    :param patch_pis: A list of correspondences between the 2D uv samples and the points in a neighborhood
    :param patch_uvs: A list of tensors, each of shape [n_i, 2] of UV positions for the given patch
    :param patch_idx: A list of tensors each of shape [n_i] containing the indices of the points in a neighborhood into
                      the input point-cloud x (of shape [n, 3])
    :param patch_tx: A list of tuples (t_i, s_i, r_i) of transformations (t_i is a translation, s_i is a scaling, and
                     r_i is a rotation matrix) which map the points in a neighborhood to a centered and whitened point
                     set
    :param phi: A list of neural networks representing the lifting function for each chart in the atlas
    :param x: A [n, 3] tensor containing the input point cloud
    :param devices: A list of devices which the models, phi, will be run on
    :param num_batches: The number of batches on which to perform the evaluation on
    :return: A list of tensors, each of shape [n_i, 3] where each tensor is the average prediction of the overlapping
             charts a the samples
    """
    num_patches = len(patch_uvs)
    batch_size = int(np.ceil(num_patches / num_batches))

    if isinstance(x, np.ndarray):
        mean_pts = torch.from_numpy(x).to(patch_uvs[0])
    elif torch.is_tensor(x):
        mean_pts = x.clone()
    else:
        raise ValueError("Invalid type for x")

    counts = torch.ones(x.shape[0], 1).to(mean_pts)
    EPS = 1e-5

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_patches)
        for i in range(start_idx, end_idx):
            dev_i = devices[i % len(devices)]
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_pis[i] = patch_pis[i].to(dev_i)
            patch_idx[i] = patch_idx[i].to(dev_i)
            patch_tx[i] = tuple(txj.to(dev_i) for txj in patch_tx[i])
                
        for i in range(start_idx, end_idx):
            translate_i, scale_i, rotate_i = patch_tx[i]

            uv_i = patch_uvs[i]
            y_i = ((phi[i](uv_i).squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i)
            pi_i = patch_pis[i]
            idx_i = patch_idx[i][pi_i]
        
            mean_pts[idx_i] += y_i.to(mean_pts)
            counts[idx_i, :] += 1
            
        for i in range(start_idx, end_idx):
            dev_i = 'cpu'
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_pis[i] = patch_pis[i].to(dev_i)
            patch_idx[i] = patch_idx[i].to(dev_i)
            patch_tx[i] = tuple(txj.to(dev_i) for txj in patch_tx[i])

    mean_pts = mean_pts / counts

    means = []
    for i in range(num_patches):
        idx_i = patch_idx[i]
        translate_i, scale_i, rotate_i = patch_tx[i]
        device_i = translate_i.device
        m_i = scale_i * (mean_pts[idx_i].to(device_i) + translate_i) @ rotate_i
        means.append(m_i)

    return means


def upsample_surface(patch_uvs, patch_tx, patch_models, devices, scale=1.0, num_samples=12, normal_samples=32,
                     num_batches=1, compute_normals=True):
    vertices = []
    normals = []
    indices = []
    num_patches = len(patch_models)
    batch_size = int(np.ceil(num_patches / num_batches))
    with torch.no_grad():
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_patches)
            for i in range(start_idx, end_idx):
                dev_i = devices[i % len(devices)]
                patch_models[i] = patch_models[i].to(dev_i)
                patch_uvs[i] = patch_uvs[i].to(dev_i)
                patch_tx[i] = tuple(txj.to(dev_i) for txj in patch_tx[i])
                
            for i in range(start_idx, end_idx):
                if (i + 1) % 10 == 0:
                    print("Upsamling %d/%d" % (i + 1, len(patch_models)))

                device = devices[i % len(devices)]

                n = num_samples
                translate_i, scale_i, rotate_i = (patch_tx[i][j].to(device) for j in range(len(patch_tx[i])))
                uv_i = utils.meshgrid_from_lloyd_ts(patch_uvs[i].cpu().numpy(), n, scale=scale).astype(np.float32)
                uv_i = torch.from_numpy(uv_i).to(patch_uvs[i])
                y_i = patch_models[i](uv_i)

                mesh_v = ((y_i.squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()
            
                if compute_normals:
                    mesh_f = utils.meshgrid_face_indices(n)
                    mesh_n = pcu.per_vertex_normals(mesh_v, mesh_f)
                    normals.append(mesh_n)

                vertices.append(mesh_v)
                indices .append(i * np.ones((mesh_v.shape[0], 1)))
            for i in range(start_idx, end_idx):
                dev_i = 'cpu'
                patch_models[i] = patch_models[i].to(dev_i)
                patch_uvs[i] = patch_uvs[i].to(dev_i)
                patch_tx[i] = tuple(txj.to(dev_i) for txj in patch_tx[i])

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    indices = np.concatenate(indices, axis=0)
    if compute_normals:
        normals = np.concatenate(normals, axis=0).astype(np.float32)
    else:
        print("Fixing normals...")
        _, normals = pcu.estimate_point_cloud_normals_knn(vertices, normal_samples)

    return vertices, normals, indices


def bisect_search_residual(tar, idx):
    compt_proj.config(tar, idx)
    _, tar_proj = compt_proj.compute()
    sgn = -1. if np.linalg.norm(tar) < np.linalg.norm(tar_proj) else 1.
    return sgn*np.linalg.norm(tar-tar_proj)
def find_loc_on_surface(tar, idx):
    # implement a bisection search algo to find the actual crossing
    # initial guess range 15/45
    lb, ub = 10., 50.
    res_lb = bisect_search_residual(lb*tar, idx)
    res_ub = bisect_search_residual(ub*tar, idx)
    while True:
        mid = lb + (ub - lb) * abs(res_lb) / (abs(res_lb) + abs(res_ub))
        res_mb = bisect_search_residual(mid*tar, idx)
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


    
def move_optimizer_to_device(optimizer, device):
    state_devices = {}
    for i, state in enumerate(optimizer.state.values()):
        for k, v in state.items():
            if torch.is_tensor(v):
                key = k + "-" + str(i)
                dev = device[key] if isinstance(device, dict) else device
                state_devices[key] = v.device
                state[k] = v.to(dev)
    return state_devices

"""
    class that compute projection to a certain patch
"""
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("radius", type=float, help="Patch radius (The parameter, r, in the paper)")
    argparser.add_argument("padding", type=float, help="Padding factor for patches (The parameter, c, in the paper)")
    argparser.add_argument("min_pts_per_patch", type=int,
                           help="Minimum number of allowed points inside a patch used to not fit to "
                                "patches with too little data")
    argparser.add_argument("--output", "-o", type=str, default="out",
                           help="Name for the output files: e.g. if you pass in --output out, the program will save "
                                "a dense upsampled point-cloud named out.ply, and a file containing reconstruction "
                                "metadata and model weights named out.pt. Default: out -- "
                                "Note: the number of points per patch in the upsampled point cloud is 64 by default "
                                "and can be set by specifying --upsamples-per-patch.")
    argparser.add_argument("--upsamples-per-patch", "-nup", type=int, default=8,
                           help="*Square root* of the number of upsamples per patch to generate in the output. i.e. if "
                                "you pass in --upsamples-per-patch 8, there will be 64 upsamples per patch.")
    argparser.add_argument("--angle-threshold", "-a", type=float, default=95.0,
                           help="Threshold (in degrees) used to discard points in "
                                "a patch whose normal is facing the wrong way.")
    argparser.add_argument("--local-epochs", "-nl", type=int, default=25,
                           help="Number of fitting iterations done for each chart to its points")
    argparser.add_argument("--global-epochs", "-ng", type=int, default=25,
                           help="Number of fitting iterations done to make each chart agree "
                                "with its neighboring charts")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3,
                           help="Step size for gradient descent.")
    argparser.add_argument("--devices", "-d", type=str, default=["cuda"], nargs="+",
                           help="A list of devices on which to partition the models for each patch. For large inputs, "
                                "reconstruction can be memory and compute intensive. Passing in multiple devices will "
                                "split the load across these. E.g. --devices cuda:0 cuda:1 cuda:2")
    # argparser.add_argument("--plot", action="store_true",
    #                        help="Plot the following intermediate states:. (1) patch neighborhoods, "
    #                             "(2) Intermediate reconstruction before global consistency step, "
    #                             "(3) Reconstruction after global consistency step. "
    #                             "This flag is useful for debugging but does not scale well to large inputs.")
    argparser.add_argument("--interpolate", action="store_true",
                           help="If set, then force all patches to agree with the input at overlapping points "
                                "(i.e. the reconstruction will try to interpolate the input point cloud). "
                                "Otherwise, we fit all patches to the average of overlapping patches at each point.")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--sinkhorn-epsilon", "-sl", type=float, default=1e-3,
                           help="The reciprocal (1/lambda) of the Sinkhorn regularization parameter.")
    argparser.add_argument("--seed", "-s", type=int, default=-1,
                           help="Random seed to use when initializing network weights. "
                                "If the seed not positive, a seed is selected at random.")
    argparser.add_argument("--exact-emd", "-e", action="store_true",
                           help="Use exact optimal transport distance instead of sinkhorn. "
                                "This will be slow and should not make a difference in the output")
    argparser.add_argument("--use-best", action="store_true",
                           help="Use the model with the lowest loss as the final result.")
    argparser.add_argument("--normal-neighborhood-size", "-ns", type=int, default=64,
                           help="Neighborhood size used to estimate the normals in the final dense point cloud. "
                                "Default: 64")
    argparser.add_argument("--save-pre-cc", action="store_true",
                           help="Save a copy of the model before the cycle consistency step")
    argparser.add_argument("--batch-size", type=int, default=-1, help="Split fitting MLPs into batches")
    args = argparser.parse_args()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "pre_cycle_consistency_model": None,
        "final_model": None,
        "patch_uvs": None,
        "patch_idx": None,
        "patch_txs": None,
        "patch_ctr": None,
        "radius": args.radius,
        "padding": args.padding,
        "min_pts_per_patch": args.min_pts_per_patch,
        "angle_threshold": args.angle_threshold,
        "interpolate": args.interpolate,
        "global_epochs": args.global_epochs,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "devices": args.devices,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils.seed_everything(args.seed),
        "batch_size": args.batch_size
    }

    scale_dict = {
        "yld-surf-d-0"   : np.array([20.6922493, 19.84395027, 18.52919006*np.sqrt(2)]),
        "yld-surf-d-1100": np.array([24.05110931, 24.7630806, 22.44108963*np.sqrt(2)]),
        "yld-surf-d-1110": np.array([23.27001953, 22.43214035, 21.24725914*np.sqrt(2)]),
        "yld-surf-d-2100": np.array([28.18310928, 29.00629997, 27.18322945*np.sqrt(2)]),
        "yld-surf-d-2110": np.array([26.17239952, 25.40192032, 23.67305946*np.sqrt(2)])
    }
    #surface_scale = scale_dict[args.output]

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=True)

    # Compute a set of neighborhood (patches) and a uv samples for each neighborhood. Store the result in a list
    # of pairs (uv_j, xi_j) where uv_j are 2D uv coordinates for the j^th patch, and xi_j are the indices into x of
    # the j^th patch. We will try to reconstruct a function phi, such that phi(uv_j) = x[xi_j].
    print("Computing neighborhoods...")
    bbox_diag = np.linalg.norm(np.max(x, axis=0) - np.min(x, axis=0))
    patch_idx, patch_uvs, patch_xs, patch_tx, ctr_v = compute_patches(x, n, args.radius*bbox_diag, args.padding,
                                                               angle_thresh=args.angle_threshold,
                                                               min_pts_per_patch=args.min_pts_per_patch)
    # create kdtree for neighbor search
    kdtree = cKDTree(ctr_v)

    num_patches = len(patch_uvs)
    
    output_dict["patch_uvs"] = patch_uvs
    output_dict["patch_idx"] = patch_idx
    output_dict["patch_xs" ] = patch_xs
    output_dict["patch_txs"] = patch_tx
    output_dict["patch_ctr"] = ctr_v


    # FIXME: Plotting is currently broken
    # if args.plot:
    #     plot_patches(x, patch_idx)

    # Initialize one model per patch and convert the input data to a pytorch tensor
    print("Creating models...")
    if args.batch_size > 0:
        num_batches = int(np.ceil(num_patches / args.batch_size))
        batch_size = args.batch_size
        print("Splitting fitting into %d batches" % num_batches)
    else:
        num_batches = 1
        batch_size = num_patches
    #phi = [MLP(2, 3) for i in range(num_patches)]
    phi = nn.ModuleList([MLP(2, 3) for i in range(num_patches)])
    # x = torch.from_numpy(x.astype(np.float32)).to(args.device)

    phi_optimizers = []
    phi_optimizers_devices = []
    uv_optimizer = torch.optim.Adam(patch_uvs, lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)
    mse_loss = nn.MSELoss()

    # Fit a function, phi_i, for each patch so that phi_i(patch_uvs[i]) = x[patch_idx[i]]. i.e. so that the function
    # phi_i "agrees" with the point cloud on each patch.
    #
    # We also store the correspondences between the uvs and points which we use later for the consistency step. The
    # correspondences are stored in a list, pi where pi[i] is a vector of integers used to permute the points in
    # a patch.
    pi = [None for _ in range(num_patches)]

    # Cache model with the lowest loss if --use-best is passed
    best_models = [None for _ in range(num_patches)]
    best_losses = [np.inf for _ in range(num_patches)]

    print("Training local patches...")
    for b in range(num_batches):
        print("Fitting batch %d/%d" % (b + 1, num_batches))
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_patches)
        optimizer_batch = torch.optim.Adam(phi[start_idx:end_idx].parameters(), lr=args.learning_rate)
        #optimizer_batch = torch.optim.Adam(phi[start_idx].parameters(), lr=args.learning_rate)
        phi_optimizers.append(optimizer_batch)
        for i in range(start_idx, end_idx):
            print("moving the {}th models to GPU ......".format(i+1))
            dev_i = args.devices[i % len(args.devices)]
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
        print("Patch data all loaded to GPU!")
            
        for epoch in range(args.local_epochs):
            optimizer_batch.zero_grad()
            uv_optimizer.zero_grad()

            # sum_loss = torch.tensor([0.0]).to(args.devices[0])
            losses = []
            torch.cuda.synchronize()
            epoch_start_time = time.time()
            for i in range(start_idx, end_idx):
                uv_i = patch_uvs[i]
                x_i = patch_xs[i]
                y_i = phi[i](uv_i)

                with torch.no_grad():
                    if args.exact_emd:
                        M_i = pairwise_distances(x_i.unsqueeze(0), y_i.unsqueeze(0)).squeeze().cpu().squeeze().numpy()
                        p_i = ot.emd(np.ones(x_i.shape[0]), np.ones(y_i.shape[0]), M_i)
                        p_i = torch.from_numpy(p_i.astype(np.float32)).to(args.devices[0])
                    else:
                        _, p_i = sinkhorn_loss(x_i.unsqueeze(0), y_i.unsqueeze(0))
                    pi_i = p_i.squeeze().max(0)[1]
                    pi[i] = pi_i

                loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

                if args.use_best and loss_i.item() < best_losses[i]:
                    best_losses[i] = loss_i.item()
                    model_copy = copy.deepcopy(phi[i]).to('cpu')
                    best_models[i] = copy.deepcopy(model_copy.state_dict())
                loss_i.backward(retain_graph=False)
                losses.append(loss_i.detach())
                # sum_loss += loss_i.to(args.devices[0])

            # sum_loss.backward()
            sum_loss = sum([l.item() for l in losses])
            torch.cuda.synchronize()
            epoch_end_time = time.time()

            print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
                  (epoch, args.local_epochs, sum_loss,
                   sum_loss / (end_idx - start_idx), epoch_end_time - epoch_start_time))
            optimizer_batch.step()
            uv_optimizer.step()
            
        for i in range(start_idx, end_idx):
            dev_i = 'cpu'
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            pi[i] = pi[i].to(dev_i)
        optimizer_batch_devices = move_optimizer_to_device(optimizer_batch, 'cpu')
        phi_optimizers_devices.append(optimizer_batch_devices)
                    
        print("Done batch %d/%d" % (b + 1, num_batches))

    print("Mean best losses:", np.mean(best_losses[i]))
    
    if args.use_best:
        for i, phi_i in enumerate(phi):
            phi_i.load_state_dict(best_models[i])

    if args.save_pre_cc:
        output_dict["pre_cycle_consistency_model"] = copy.deepcopy(phi.state_dict())

    # FIXME: Plotting is broken
    # if args.plot:
    #     raise NotImplementedError("TODO: Fix plotting code")
    #     plot_reconstruction(x, patch_uvs, patch_tx, phi, scale=1.0/args.padding)

    # Do a second, global, stage of fitting where we ask all patches to agree with each other on overlapping points.
    # If the user passed --interpolate, we ask that the patches agree on the original input points, otherwise we ask
    # that they agree on the average of predictions from patches overlapping a given point.
    if not args.interpolate:
        print("Computing patch means...")
        with torch.no_grad():
            #print(x.shape)
            patch_xs = patch_means(pi, patch_uvs, patch_idx, patch_tx, phi, x, args.devices, num_batches)

    print("Training cycle consistency...")
    for b in range(num_batches):
        print("Fitting batch %d/%d" % (b + 1, num_batches))
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_patches)
        for i in range(start_idx, end_idx):
            dev_i = args.devices[i % len(args.devices)]
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            pi[i] = pi[i].to(dev_i)
        optimizer = phi_optimizers[b]
        move_optimizer_to_device(optimizer, phi_optimizers_devices[b])
        for epoch in range(args.global_epochs):
            optimizer.zero_grad()
            uv_optimizer.zero_grad()

            sum_loss = torch.tensor([0.0]).to(args.devices[0])
            epoch_start_time = time.time()
            for i in range(start_idx, end_idx):
                uv_i = patch_uvs[i]
                x_i = patch_xs[i]
                y_i = phi[i](uv_i)
                pi_i = pi[i]
                loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

                if loss_i.item() < best_losses[i]:
                    best_losses[i] = loss_i.item()
                    model_copy = copy.deepcopy(phi[i]).to('cpu')
                    best_models[i] = copy.deepcopy(model_copy.state_dict())

                sum_loss += loss_i.to(args.devices[0])

            sum_loss.backward()
            epoch_end_time = time.time()

            print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
                  (epoch, args.global_epochs, sum_loss.item(),
                   sum_loss.item() / (end_idx - start_idx), epoch_end_time-epoch_start_time))
            optimizer.step()
            uv_optimizer.step()
        for i in range(start_idx, end_idx):
            dev_i = 'cpu'
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            pi[i] = pi[i].to(dev_i)
        move_optimizer_to_device(optimizer, 'cpu')
                    
    print("Mean best losses:", np.mean(best_losses[i]))
    for i, phi_i in enumerate(phi):
        phi_i.load_state_dict(best_models[i])

    output_dict["patch_uvs"] = patch_uvs
    output_dict["patch_idx"] = patch_idx
    output_dict["patch_xs" ] = patch_xs
    output_dict["patch_txs"] = patch_tx
    output_dict["patch_ctr"] = ctr_v

    output_dict["final_model"] = phi.state_dict()
    """
    print("generating dense point cloud give directions...")
    pres_locs = np.loadtxt('stress_nu_12.txt')
    Ndir = pres_locs.shape[0]
    #Ndir = 1
    gp_points_result = np.zeros((Ndir,3))
    compt_proj = Compute_project(phi, patch_tx, surface_scale)
    for i in range(Ndir):
        tar_loc = pres_locs[i,:]
        ctr_dist = np.arccos( (np.dot(ctr_v,tar_loc)) / np.linalg.norm(ctr_v,axis=1) )
        idxes = np.argsort(ctr_dist)
        temp_points = [find_loc_on_surface(tar_loc, idxes[k]) for k in range(5)]
        temp_points_norm = [np.dot(tar_loc, p) / np.linalg.norm(p) for p in temp_points]
        gp_points_result[i,:] = temp_points[np.argmax(temp_points_norm)]
        print('find solution for projection problem:\npred = {}\ntar dir = {}'.format(gp_points_result[i,:], tar_loc))
    np.savetxt(args.output+'-gp-points.txt', gp_points_result, fmt='%.6f %.6f %.6f')
    """
    print("Generating dense point cloud...")
    v, n, i_ = upsample_surface(patch_uvs, patch_tx, phi, args.devices,
                            scale=0.9, #(1.0/args.padding)
                            num_samples=args.upsamples_per_patch,
                            normal_samples=args.normal_neighborhood_size,
                            num_batches=num_batches,
                            compute_normals=False)

    #print("Saving dense point cloud...")
    #pcu.write_ply(args.output + ".ply", v, np.zeros([], dtype=np.int32), n, np.zeros([], dtype=v.dtype))
    """ new edit """
    #np.save(args.output+".npy", np.concatenate((v, n), axis=1))
    np.savetxt(args.output+"_vertices.txt", v, fmt='%.8g %.8g %.8g')
    np.savetxt(args.output+"_normals.txt", n, fmt='%.8g %.8g %.8g')
    np.savetxt(args.output+"_indices.txt", i_, fmt='%d')
    print("Saving metadata...")
    torch.save(output_dict, args.output + ".pt")










