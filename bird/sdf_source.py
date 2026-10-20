"""Exact signed-distance field of the source sculpt, for smooth resampling.

Distances are exact (point-to-triangle, libigl) in a band around the surface and
interpolated from a coarse exact grid further away. Sign comes from the generalized
winding number, which copes with the sculpt's buried inner shells.
"""
import numpy as np
import igl
import trimesh
from scipy import ndimage


class SDFSource:
    def __init__(self, field, lo, pitch):
        self.field, self.lo, self.pitch = field, np.asarray(lo, float), float(pitch)

    def __call__(self, P):
        idx = ((P - self.lo) / self.pitch).reshape(-1, 3).T
        return ndimage.map_coordinates(self.field, idx, order=1, cval=10.0,
                                       mode="constant").reshape(P.shape[:-1])

    def save(self, path):
        np.savez(path, field=self.field, lo=self.lo, pitch=self.pitch)

    @staticmethod
    def load(path):
        d = np.load(path)
        return SDFSource(d["field"], d["lo"], float(d["pitch"]))


def clean_mesh(mesh):
    """Consistent outward winding, and without faces buried inside the solid."""
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    trimesh.repair.fix_normals(m, multibody=True)
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    C, N = m.triangles_center, m.face_normals
    w_out = igl.fast_winding_number(V, F, C + 0.03 * N)
    w_in = igl.fast_winding_number(V, F, C - 0.03 * N)
    buried = (w_out > 0.5) & (w_in > 0.5)
    m.update_faces(~buried)
    m.remove_unreferenced_vertices()
    return m, int(buried.sum())


def build(mesh, pitch=0.25, coarse=1.0, band=1.5, margin=3.0):
    V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    lo = mesh.bounds[0] - margin
    hi = mesh.bounds[1] + margin

    def sd(P):
        out = np.empty(len(P))
        for i in range(0, len(P), 2_000_000):
            out[i:i + 2_000_000] = igl.signed_distance(
                P[i:i + 2_000_000], V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
        return out

    # coarse exact grid
    cshape = tuple(np.ceil((hi - lo) / coarse).astype(int) + 1)
    axes = [lo[i] + coarse * np.arange(cshape[i]) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    cfield = sd(G).reshape(cshape).astype(np.float32)

    # fine grid: interpolate, then exact values near the surface
    shape = tuple(np.ceil((hi - lo) / pitch).astype(int) + 1)
    field = np.empty(shape, np.float32)
    ys = np.arange(shape[1]) * pitch
    zs = np.arange(shape[2]) * pitch
    for i in range(shape[0]):
        x = i * pitch
        I = np.full((shape[1], shape[2]), x / coarse)
        J, K = np.meshgrid(ys / coarse, zs / coarse, indexing="ij")
        field[i] = ndimage.map_coordinates(cfield, [I, J, K], order=1, mode="nearest")
    near = np.abs(field) < band + coarse
    idx = np.argwhere(near)
    P = lo + idx * pitch
    field[near] = sd(P).astype(np.float32)
    return SDFSource(field, lo, pitch), int(near.sum())
