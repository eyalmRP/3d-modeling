"""Color rules for the re-posed sculpt.

Every surface point is traced back to its piece (body, head, tail, neck) and to its
position on the original sculpt, then colored with rules written in that piece's own
frame, following the reference sheet: purple body and head, white chest/throat, navy
and white wing and tail tips, red beak with yellow patches, white eyes with black
pupils, light-blue lids, black brows, orange legs and feet.
"""
import numpy as np
from matplotlib.path import Path

import stl_repose as R

PALETTE = {
    "purple": (0.36, 0.33, 0.64),
    "navy": (0.09, 0.08, 0.30),
    "white": (0.95, 0.95, 0.93),
    "red": (0.86, 0.05, 0.07),
    "yellow": (0.98, 0.66, 0.06),
    "orange": (0.80, 0.17, 0.05),
    "blue": (0.50, 0.66, 0.94),
    "black": (0.03, 0.03, 0.04),
}
NAMES = list(PALETTE)
ID = {n: i for i, n in enumerate(NAMES)}

# eyeballs fitted on the sculpt, head-local frame (x across, -y forward, z up)
EYES = {+1: (np.array([9.36, 4.32, 5.02]), 10.61), -1: (np.array([-10.25, 4.58, 4.79]), 10.84)}

# folded wing outline in the side view (y, z), from the shoulder round to the tip
WING = [(-35.7, 51.6), (-28.3, 52.0), (-20.6, 49.7), (-14.3, 46.6), (-8.9, 41.9), (-3.4, 37.2),
        (2.0, 31.0), (6.7, 24.8), (9.8, 19.3), (12.6, 13.2), (8.2, 15.8), (1.2, 20.1),
        (-6.6, 25.6), (-14.3, 29.1), (-22.1, 32.2), (-29.9, 36.8), (-34.2, 43.4), (-36.5, 49.0)]
WING_A, WING_B = np.array([-34.0, 50.0]), np.array([12.1, 13.5])   # shoulder -> tip


def head_local(S):
    f, l, u = R.head_frame()
    d = S - R.HEAD_C
    return np.stack([d @ l, -(d @ f), d @ u], -1)


def paint_head(H, N):
    """H: head-local points, N: matching unit normals."""
    x, y, z = H[:, 0], H[:, 1], H[:, 2]
    out = np.full(len(H), ID["purple"])
    side = np.where(x >= 0, 1, -1)
    ec = np.where(side[:, None] > 0, EYES[1][0], EYES[-1][0])
    er = np.where(side > 0, EYES[1][1], EYES[-1][1])
    rad = H - ec
    dist = np.linalg.norm(rad, axis=1)
    de = dist - er
    radial = np.einsum("ij,ij->i", rad / dist[:, None], N)
    on_eye = (np.abs(de) < 0.4) & (radial > 0.85) & (side * x > 3.0) & (y - ec[:, 1] < 0.35 * er)

    beak = (y < -4.0) & ~on_eye
    beak |= (y < 1.0) & (z > 13.0) & (np.abs(x) < 5.0)              # culmen runs up the forehead
    jaw = (z < -1.5) & (z > -12.0) & (y < 6.0) & (np.abs(x) > 7.0) & (de > 1.0)   # smile corners
    out[beak | jaw] = ID["red"]
    yellow = beak & (np.abs(x) > 2.8) & (z > 1.5) & (((y + 9.0) / 14.0) ** 2 + ((z - 10.5) / 8.5) ** 2 < 1.0)
    out[yellow] = ID["yellow"]

    throat = (z < -9.0) & (y < -2.0) & (np.abs(x) < 10.0) & ~beak & ~jaw
    out[throat] = ID["white"]

    lids = (de > 0.0) & (de < 2.4) & ~on_eye & ~beak & ~jaw & (y < ec[:, 1] + 2.0) & (side * x > 2.0)
    out[lids] = ID["blue"]
    out[on_eye] = ID["white"]
    # heavy, half-closed upper lid over the top of each eyeball (the sheet's sleepy look)
    n_lid = np.array([0.0, 0.22, 1.0]) / np.linalg.norm([0.0, 0.22, 1.0])
    lid_cap = on_eye & ((rad @ n_lid) > 0.22 * er)
    # pupils: both glance up and forward, tucked under the lid
    g = np.stack([side * 0.60, -0.70 * np.ones_like(x), 0.20 * np.ones_like(x)], 1)
    g /= np.linalg.norm(g, axis=1)[:, None]
    cosang = np.einsum("ij,ij->i", rad / dist[:, None], g)
    out[on_eye & (cosang > np.cos(np.radians(19)))] = ID["black"]
    out[lid_cap] = ID["blue"]

    tuft = (y > 16.0) & (np.linalg.norm(H, axis=1) > 20.5) & (np.abs(x) < 12.0) & (z < 6.0)
    out[tuft] = ID["navy"]
    brows = (z > 12.5) & (np.abs(x) > 4.0) & (np.abs(x) < 23.0) & (y > -7.0) & (y < 19.0) & (de > 2.4) & ~beak
    out[brows] = ID["black"]
    return out


# chest and belly seen from the side: in front of / below the folded wing
WHITE_SIDE = [(-31.0, 62.0), (-44.0, 50.0), (-44.0, 25.0), (-35.0, 10.0), (0.0, 8.0), (20.0, 10.0),
              (20.0, 17.0), (12.6, 13.2), (8.2, 15.8), (1.2, 20.1), (-6.6, 25.6), (-14.3, 29.1),
              (-22.1, 32.2), (-29.9, 36.8), (-34.2, 43.4), (-36.5, 49.0), (-35.7, 51.6), (-26.0, 58.0)]


WHITE_FRONT = [(-31.0, 62.0), (-44.0, 50.0), (-40.0, 40.0), (-30.0, 45.0), (-24.0, 52.0), (-18.0, 60.0)]


def paint_body(S):
    x, y, z = np.abs(S[:, 0]), S[:, 1], S[:, 2]
    yz = np.c_[y, z]
    out = np.full(len(S), ID["purple"])
    out[Path(WHITE_SIDE).contains_points(yz)] = ID["white"]
    out[(x < 9.0) & Path(WHITE_FRONT).contains_points(yz)] = ID["white"]   # up the front to the neck
    wing = Path(WING).contains_points(yz) & (x > 6.0)
    ax = (WING_B - WING_A) / np.linalg.norm(WING_B - WING_A)
    t = (yz - WING_A) @ ax / np.linalg.norm(WING_B - WING_A)
    across = (yz - WING_A) @ np.array([-ax[1], ax[0]])
    t1 = 0.44 + 0.035 * np.abs(np.cos(across * np.pi / 4.5))      # scalloped white row
    out[wing] = ID["purple"]
    out[wing & (t > 0.37) & (t <= t1)] = ID["white"]
    out[wing & (t > t1)] = ID["navy"]
    out[z < 15.5] = ID["orange"]                                     # legs and feet
    return out


def paint_tail(S):
    a = R.unit(R.TAIL_AXIS)
    t = (S - R.TAIL_ROOT) @ a / 128.0
    out = np.full(len(S), ID["purple"])
    out[t > 0.60] = ID["white"]
    out[t > 0.70] = ID["navy"]
    return out


def paint_neck(P, pose):
    a = pose.neck_pts[0]
    out = np.full(len(P), ID["purple"])
    out[P[:, 1] < a[1] + 0.5 + (P[:, 2] - a[2]) * 0.55] = ID["white"]
    return out


class SourceLabels:
    """Colors computed on the original sculpt (smooth surface and normals), looked up by
    nearest source vertex for each piece."""

    def __init__(self, src_mesh, src):
        V = np.asarray(src_mesh.vertices, float)
        N = np.asarray(src_mesh.vertex_normals, float)
        outer = src(V) < 0.9                     # drop hidden inner shells (e.g. buried eyeballs)
        V, N = V[outer], N[outer]
        self.trees, self.labels = {}, {}
        sel = R.head_region(V)
        f, l, u = R.head_frame()
        Nh = np.stack([N[sel] @ l, -(N[sel] @ f), N[sel] @ u], -1)
        self._add("head", V[sel], paint_head(head_local(V[sel]), Nh))
        sel = (V[:, 0] > -1.0) & ~R.head_region(V)
        self._add("body", V[sel], paint_body(V[sel]))
        sel = R.tail_region(V)
        self._add("tail", V[sel], paint_tail(V[sel]))

    def _add(self, name, pts, labels):
        from scipy.spatial import cKDTree
        self.trees[name] = cKDTree(pts)
        self.labels[name] = labels

    def lookup(self, name, S):
        _, i = self.trees[name].query(S)
        return self.labels[name][i]


def smooth_labels(mesh, labels, iterations=3):
    """Majority vote over each vertex and its neighbours, to remove isolated specks."""
    import scipy.sparse as sp
    n = len(mesh.vertices)
    e = mesh.edges_unique
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr() + sp.identity(n, format="csr")
    for _ in range(iterations):
        onehot = sp.coo_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, len(NAMES))).tocsr()
        labels = np.asarray((A @ onehot).argmax(1)).ravel()
    return labels


def merge_islands(mesh, labels, min_area=4.0, rounds=3):
    """Reassign color islands smaller than min_area (mm^2 of surface) to their neighbours."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = len(mesh.vertices)
    e = mesh.edges_unique
    varea = np.zeros(n)
    np.add.at(varea, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3.0, 3))
    for _ in range(rounds):
        same = labels[e[:, 0]] == labels[e[:, 1]]
        g = sp.coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n))
        ncomp, comp = connected_components(g, directed=False)
        area = np.bincount(comp, weights=varea, minlength=ncomp)
        small = area[comp] < min_area
        if not small.any():
            break
        # vote among neighbours that are not in a small island
        votes = np.zeros((n, len(NAMES)))
        for a, b in ((e[:, 0], e[:, 1]), (e[:, 1], e[:, 0])):
            ok = small[a] & ~small[b]
            np.add.at(votes, (a[ok], labels[b[ok]]), 1)
        has = votes.sum(1) > 0
        labels = labels.copy()
        labels[small & has] = votes[small & has].argmax(1)
    return labels


def paint(mesh, src, pose, pitch, src_labels):
    P = np.asarray(mesh.vertices, float)
    pieces = pose.pieces(src, P, pitch)
    names = list(pieces)
    owner = np.argmax(np.stack([pieces[n] for n in names]), 0)
    labels = np.zeros(len(P), int)
    for i, n in enumerate(names):
        sel = owner == i
        if not sel.any():
            continue
        if n == "head":
            labels[sel] = src_labels.lookup("head", pose.head_src(P[sel]))
        elif n == "body":
            labels[sel] = src_labels.lookup("body", pose.body_src(P[sel]))
        elif n == "tail":
            labels[sel] = src_labels.lookup("tail", pose.tail_src(P[sel]))
        else:
            labels[sel] = paint_neck(P[sel], pose)
    return merge_islands(mesh, smooth_labels(mesh, labels)), owner, names
