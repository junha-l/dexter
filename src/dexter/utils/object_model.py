"""
Object model for DexGYS dataset using csdf for SDF computation.
"""

import os

import torch
import trimesh as tm
from csdf import compute_sdf, index_vertices_by_faces


def farthest_point_sample(points, num_samples):
    """Farthest point sampling, seeded at index 0.

    Equivalent to pytorch3d.ops.sample_farthest_points with random_start_point=False,
    which this replaces so the package does not need to be installed.

    Parameters
    ----------
    points: (N, 3) torch.Tensor
    num_samples: int

    Returns
    -------
    (num_samples, 3) torch.Tensor
    """
    n = points.shape[0]
    if num_samples >= n:
        return points
    selected = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    closest = torch.full((n,), float("inf"), device=points.device)
    farthest = torch.zeros((), dtype=torch.long, device=points.device)
    for i in range(num_samples):
        selected[i] = farthest
        closest = torch.minimum(closest, (points - points[farthest]).pow(2).sum(-1))
        farthest = closest.argmax()
    return points[selected]


class ObjectModel:
    def __init__(self, data_root_path, num_samples=2000, device="cuda"):
        """
        Parameters
        ----------
        data_root_path: str
            DexGYS dataset root (expects data/{obj_id}/urdf/meshes/decomposed.obj)
        num_samples: int
            Number of object surface points sampled with FPS
        device: str | torch.device
        """
        self.device = device
        self.data_root_path = data_root_path
        self.num_samples = num_samples
        self.batch_size_each = None
        self.object_code_list = None
        self.object_mesh_list = None
        self.object_face_verts_list = None

    def initialize(self, object_code_list):
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []

        for object_code in object_code_list:
            mesh_path = os.path.join(
                self.data_root_path, "data", object_code, "urdf", "meshes", "decomposed.obj"
            )
            self.object_mesh_list.append(tm.load(mesh_path, force="mesh", process=False))
            object_verts = torch.tensor(
                self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device
            )
            object_faces = torch.tensor(
                self.object_mesh_list[-1].faces, dtype=torch.long, device=self.device
            )
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))

            if self.num_samples != 0:
                # Area-weighted dense sample, then FPS down to num_samples for even
                # surface coverage.
                dense_point_cloud = torch.tensor(
                    self.object_mesh_list[-1].sample(100 * self.num_samples),
                    dtype=torch.float,
                    device=self.device,
                )
                surface_points = farthest_point_sample(dense_point_cloud, self.num_samples)
                self.surface_points_tensor.append(surface_points)

        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(
                self.surface_points_tensor, dim=0
            ).repeat_interleave(
                self.batch_size_each, dim=0
            )  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from points to object meshes.
        Interiors are positive, exteriors are negative.

        Parameters
        ----------
        x: (B, n_contact, 3) torch.Tensor
        with_closest_points: bool

        Returns
        -------
        distance: (B, n_contact) torch.Tensor  — interior positive
        normals: (B, n_contact, 3) torch.Tensor
        closest_points: (B, n_contact, 3) torch.Tensor  — only when with_closest_points=True
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points_list = []

        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            # csdf returns: (squared_dist, normal, dist_sign, min_dist_idx, dist_type)
            # dist_sign: -1 = inside, +1 = outside
            dis, normal, dis_signs, _, _ = compute_sdf(x[i], face_verts)
            dis_signs = dis_signs.float()
            if with_closest_points:
                closest_points_list.append(x[i] - torch.sqrt(dis).unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)  # interior positive, exterior negative
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))

        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)

        if with_closest_points:
            closest_points_out = torch.stack(closest_points_list).reshape(-1, n_points, 3)
            return distance, normals, closest_points_out
        return distance, normals
