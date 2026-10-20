"""Re-pose the sculpted bird STL to match the reference sheet.

The source sculpt has the head turned back over its shoulder, one wing raised in
front of the chest and a very long tail swung to one side. Rather than bend the
mesh, it is voxelized into a solid and cut into rigid pieces:

* body  - the half carrying the folded wing, mirrored across the body's own
          symmetry plane (this also removes the raised wing and evens the legs)
* head  - rotated as a rigid piece to face forward, re-attached with a new neck
* tail  - re-aimed into the symmetry plane and shortened

Everything is resampled into one grid, so the result is a single watertight solid.
Units are mm; the output frame has Z up, the bird facing -Y, symmetry plane x = 0.
"""
import numpy as np
import trimesh
from scipy import ndimage


def voxelize(mesh, pitch, margin=3.0):
    """Solid occupancy of a closed mesh: dense surface samples, then fill the inside."""
    lo = mesh.bounds[0] - margin
    hi = mesh.bounds[1] + margin
    shape = tuple(np.ceil((hi - lo) / pitch).astype(int) + 1)
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=pitch * 0.55)
    idx = np.rint((v - lo) / pitch).astype(int)
    shell = np.zeros(shape, bool)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = ndimage.binary_fill_holes(shell)
    return occ, lo


def grid_points(lo, pitch, shape):
    return [lo[i] + pitch * np.arange(shape[i]) for i in range(3)]


def mirror_score(occ, lo, pitch, z_range, angle_deg, offset):
    """Overlap between the occupancy in a z slab and its mirror image across the vertical
    plane through (offset along the plane normal) whose normal is at angle_deg in XY."""
    k0, k1 = [int(round((z - lo[2]) / pitch)) for z in z_range]
    slab = occ[:, :, k0:k1]
    pts = np.argwhere(slab).astype(float) * pitch
    pts[:, 0] += lo[0]
    pts[:, 1] += lo[1]
    a = np.radians(angle_deg)
    n = np.array([np.cos(a), np.sin(a)])
    d = pts[:, :2] @ n - offset
    mir = pts[:, :2] - 2 * d[:, None] * n[None, :]
    ij = np.rint((mir - lo[:2]) / pitch).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < occ.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < occ.shape[1])
    kk = np.rint(pts[:, 2] / pitch).astype(int)
    hit = np.zeros(len(pts), bool)
    hit[ok] = slab[ij[ok, 0], ij[ok, 1], kk[ok]]
    return hit.mean()


def fit_symmetry(occ, lo, pitch, z_range=(12.0, 30.0)):
    best = (-1, None, None)
    for ang in np.arange(10.0, 60.0, 1.0):
        n = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        for off in np.arange(-5.0, 25.0, 0.5):
            s = mirror_score(occ, lo, pitch, z_range, ang, off)
            if s > best[0]:
                best = (s, ang, off)
    s, ang, off = best
    # refine
    for ang2 in np.arange(ang - 1, ang + 1.01, 0.25):
        for off2 in np.arange(off - 0.5, off + 0.51, 0.125):
            s2 = mirror_score(occ, lo, pitch, z_range, ang2, off2)
            if s2 > best[0]:
                best = (s2, ang2, off2)
    return best


# symmetry plane fitted on the lower torso (normal angle in XY, offset along the normal)
PLANE_ANGLE, PLANE_OFFSET = 48.75, 7.4


def body_frame_matrix(angle=PLANE_ANGLE, offset=PLANE_OFFSET):
    """4x4 transform from the STL frame to the body frame (x across, -y forward, z up)."""
    a = np.radians(angle)
    n = np.array([np.cos(a), np.sin(a), 0.0])       # lateral, towards the folded wing
    f = np.array([n[1], -n[0], 0.0])                 # forward (the chest side)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = n, -f, [0, 0, 1]
    M[0, 3] = -offset
    return M


def head_region(p):
    """Head (with beak, brows and cheek tufts) in the body frame of the source sculpt:
    above the neck cut there is only the head, the raised wing's fingertips and the tail."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    far = np.linalg.norm(p - HEAD_C, axis=-1) > 27.0
    raised_wing = (x < -18.0) & (z < 70.0) & far
    return (z > 59.0) & (y < 12.0) & ~raised_wing


def fit_plane_points(pts, normal0, offset0):
    """Symmetry plane of a point set: minimise the mean distance between the points
    and their mirror images. Returns (unit normal, offset) with n.p = offset on the plane."""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    sub = pts[:: max(1, len(pts) // 6000)]

    def unpack(q):
        az, el, off = q
        n = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return n, off

    def cost(q):
        n, off = unpack(q)
        d = sub @ n - off
        mir = sub - 2 * d[:, None] * n[None, :]
        dist, _ = tree.query(mir)
        return np.mean(np.minimum(dist, 3.0))

    n0 = np.asarray(normal0, float) / np.linalg.norm(normal0)
    q0 = [np.arctan2(n0[1], n0[0]), np.arcsin(n0[2]), offset0]
    best = None
    for daz in (-0.3, 0.0, 0.3):
        for del_ in (-0.25, 0.0, 0.25):
            r = minimize(cost, [q0[0] + daz, q0[1] + del_, q0[2]], method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-5, maxiter=600))
            if best is None or r.fun < best.fun:
                best = r
    n, off = unpack(best.x)
    return n, off, best.fun


# ------------------------------------------------------------------ assembly

HEAD_C = np.array([-3.01, -21.8, 75.94])          # cranium centre (source, body frame)
HEAD_L = np.array([-0.374, 0.927, 0.029])          # head symmetry-plane normal
BEAK_TIP = np.array([51.6, -0.9, 89.7])
NECK_TOP = np.array([-2.3, -22.0, 59.0])           # neck centre where the head is cut off

TAIL_ROOT = np.array([1.0, 22.0, 27.0])
ANKLE = (17.0, -10.0)                              # right ankle axis (source, body frame)
TAIL_AXIS = np.array([0.069, 0.744, 0.665])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def head_frame():
    l = unit(HEAD_L)
    b = BEAK_TIP - HEAD_C
    f = unit(b - (b @ l) * l)
    u = np.cross(f, l)
    return f, l, u


def head_rotation(pitch_deg):
    """Rotation turning the head to face -Y with the beak pitched down by pitch_deg."""
    f, l, u = head_frame()
    b = np.radians(pitch_deg)
    ft = np.array([0.0, -np.cos(b), -np.sin(b)])
    lt = np.array([1.0, 0.0, 0.0])
    ut = np.cross(ft, lt)
    return np.c_[ft, lt, ut] @ np.c_[f, l, u].T


def rot_between(a, b):
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = a @ b
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


# front of the chest on the midline (z, y), measured on the source below the raised wing
CHEST_FRONT = ((24, -29.5), (30, -33.0), (34, -35.5), (38, -37.5), (42, -37.5), (44, -36.0),
               (46, -34.8), (50, -33.5), (54, -31.5), (58, -31.0))


def raised_wing_scraps(p):
    """Bits of the raised wing that cross to the kept side, just in front of the chest."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    zf, yf = np.array(CHEST_FRONT).T
    front = np.interp(z, zf, yf)
    return (x < 12.0) & (z > 26.0) & (z < 60.0) & (y < front - 0.5)


def rot_about(axis, angle):
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def tail_region(p):
    d = (p - TAIL_ROOT) @ unit(TAIL_AXIS)
    x, z = p[..., 0], p[..., 2]
    y = p[..., 1]
    return (d > 0) & (z > 22.0) & (y > 14.0) & ((np.abs(x) < 8.0) | (d > 12.0))


def capsule_sdf(P, a, b, r1, r2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    q = a + t[..., None] * ab
    return np.linalg.norm(P - q, axis=-1) - (r1 + (r2 - r1) * t)


class Source:
    """Smoothed occupancy of the source sculpt (body frame) with trilinear lookup."""

    def __init__(self, mesh, pitch, sigma=0.7):
        occ, lo = voxelize(mesh, pitch)
        self.field = ndimage.gaussian_filter(occ.astype(np.float32), sigma)
        self.lo, self.pitch = lo, pitch

    def __call__(self, P):
        idx = ((P - self.lo) / self.pitch).reshape(-1, 3).T
        return ndimage.map_coordinates(self.field, idx, order=1, cval=0.0).reshape(P.shape[:-1])


class Pose:
    """The re-pose transforms: maps any target point back to its piece and source point."""

    def __init__(self, params):
        self.p = params
        self.tilt = np.radians(params.get("body_tilt", 0.0))       # lean the upper body back
        self.pivot = np.array([0.0, *params.get("body_pivot", (-5.0, 21.0))])
        self.sh = params.get("head_scale", 1.0)
        self.legs = params.get("leg_stretch", (0.0, 6.0, 16.0))      # (extra mm, from z, to z)
        neck_base = self.body_fwd(np.array([0.0, NECK_TOP[1] - 1.5, 51.0]))
        self.Rh = head_rotation(params["head_pitch"])
        socket_off = self.sh * (self.Rh @ (NECK_TOP - HEAD_C))
        # the head sits so that its socket lands at neck_base + neck_offset
        self.socket = neck_base + np.asarray(params["neck_offset"], float)
        self.c_new = self.socket - socket_off
        via = [neck_base + np.asarray(q, float) for q in params.get("neck_via", [])]
        self.neck_pts = [neck_base, *via, self.socket]
        self.Rt = rot_between(TAIL_AXIS, unit(params["tail_axis"]))
        self.root_new = self.body_fwd(np.asarray(params["tail_root"], float))
        roll = np.radians(params.get("tail_roll", 0.0))
        self.Rroll = rot_about(unit(params["tail_axis"]), roll)
        self.joints = [(neck_base + [0, 0, 3.0], 11.0), (self.socket, 11.0), (self.root_new, 10.0),
                       (self.body_fwd(np.array([0.0, -36.0, 46.5])), 7.0)]

    def body_fwd(self, p):
        """Where a source point of the upper body ends up after the tilt and leg stretch."""
        t = self.tilt
        d = p - self.pivot
        q = self.pivot + np.array([d[0], d[1] * np.cos(t) + d[2] * np.sin(t),
                                   -d[1] * np.sin(t) + d[2] * np.cos(t)])
        return q + [0.0, 0.0, self.legs[0]]

    def unstretch(self, z):
        extra, za, zb = self.legs
        if not extra:
            return z
        return np.where(z <= za, z, np.where(z >= zb + extra, z - extra,
                                             za + (z - za) * (zb - za) / (zb + extra - za)))

    def body_src(self, P):
        S = P.copy()
        S[..., 2] = self.unstretch(S[..., 2])
        if self.tilt:                                   # undo the lean, blended in over the hips
            w = np.clip((S[..., 2] - (self.pivot[2] - 3.0)) / 6.0, 0.0, 1.0)
            t = -self.tilt * w
            dy, dz = S[..., 1] - self.pivot[1], S[..., 2] - self.pivot[2]
            S[..., 1] = self.pivot[1] + dy * np.cos(t) + dz * np.sin(t)
            S[..., 2] = self.pivot[2] - dy * np.sin(t) + dz * np.cos(t)
        S[..., 0] = np.abs(S[..., 0])
        narrow = self.p.get("leg_narrow", 0.0)           # bring the legs in towards the midline
        if narrow:
            S[..., 0] += narrow * np.clip((24.0 - S[..., 2]) / 14.0, 0.0, 1.0)
        twist = np.radians(self.p.get("foot_twist", 0.0))
        if twist:
            ax, ay = ANKLE
            t = twist * np.clip((10.0 - S[..., 2]) / 3.0, 0.0, 1.0)
            dx, dy = S[..., 0] - ax, S[..., 1] - ay
            c, s_ = np.cos(-t), np.sin(-t)
            S[..., 0], S[..., 1] = ax + c * dx - s_ * dy, ay + s_ * dx + c * dy
        shift = self.p.get("foot_shift", 0.0)          # slide the feet back, leaning the shins
        if shift:
            za, zb = self.p.get("foot_shift_zone", (4.0, 21.0))
            S[..., 1] -= shift * np.clip((zb - S[..., 2]) / (zb - za), 0.0, 1.0)
        return S

    def head_src(self, P):
        S = ((P - self.c_new) @ self.Rh) / self.sh + HEAD_C
        sb, bend = self.p.get("beak_scale", 1.0), self.p.get("beak_bend", 0.0)
        if sb == 1.0 and not bend:
            return S
        f, l, u = head_frame()
        d = S - HEAD_C
        hx, hy, hz = d @ l, -(d @ f), d @ u               # head-local, -y forward
        y0, L = -4.0, 56.0
        ahead = np.maximum(y0 - hy, 0.0)                   # how far out along the beak
        hy_src = np.where(hy < y0, y0 - ahead / sb, hy)
        hz_src = hz + bend * (np.maximum(y0 - hy_src, 0.0) / L) ** 2
        return HEAD_C + hx[..., None] * l - hy_src[..., None] * f + hz_src[..., None] * u

    def tail_src(self, P):
        s_len, s_wid = self.p["tail_scale"]
        a = unit(TAIL_AXIS)
        Q = ((P - self.root_new) @ self.Rroll) @ self.Rt
        Q = Q + ((1.0 / s_len) - 1.0) * (Q @ a)[..., None] * a
        Q = Q - (1.0 - 1.0 / s_wid) * (Q - (Q @ a)[..., None] * a)
        return Q + TAIL_ROOT

    def neck_sdf(self, P):
        d = np.full(P.shape[:-1], 1e9)
        r = np.atleast_1d(self.p["neck_radius"]).astype(float)
        radii = np.interp(np.linspace(0, 1, len(self.neck_pts)), np.linspace(0, 1, len(r)), r)
        for i, (a, b) in enumerate(zip(self.neck_pts[:-1], self.neck_pts[1:])):
            d = np.minimum(d, capsule_sdf(P, a, b, radii[i], radii[i + 1]))
        return d

    def pieces(self, src, P, pitch):
        """Occupancy-like field of each piece at points P (any shape [..., 3])."""
        S = self.body_src(P)
        body = src(S)
        body[(S[..., 2] > self.p.get("body_top", 57.0)) & ~tail_region(S)] = 0.0
        body[tail_region(S) | raised_wing_scraps(S)] = 0.0
        body[(S[..., 2] < 11.0) & (S[..., 0] < 6.0)] = 0.0    # other foot's toes near the midline
        S = self.head_src(P)
        head = src(S)
        head[~head_region(S)] = 0.0
        S = self.tail_src(P)
        tail = src(S)
        tail[~tail_region(S)] = 0.0
        neck = np.clip(0.5 - self.neck_sdf(P) / (2 * pitch * 1.5), 0, 1).astype(np.float32)
        return {"body": body, "head": head, "tail": tail, "neck": neck}


def assemble(src, params, pitch, bounds, separate=False):
    """Return the union field on the target grid plus the grid axes
    (or one field per piece when separate=True)."""
    (x0, x1), (y0, y1), (z0, z1) = bounds
    xs = np.arange(x0, x1 + pitch, pitch, dtype=np.float32)
    ys = np.arange(y0, y1 + pitch, pitch, dtype=np.float32)
    zs = np.arange(z0, z1 + pitch, pitch, dtype=np.float32)
    pose = Pose(params)
    names = ("body", "head", "tail", "neck")
    outs = {n: np.zeros((len(xs), len(ys), len(zs)), np.float32) for n in names}
    step = max(1, int(3e6 // (len(xs) * len(ys))))
    for k in range(0, len(zs), step):
        X, Y, Z = np.meshgrid(xs, ys, zs[k:k + step], indexing="ij")
        P = np.stack([X, Y, Z], -1).astype(np.float64)
        for n, f in pose.pieces(src, P, pitch).items():
            outs[n][:, :, k:k + step] = f
    axes = (xs, ys, zs)
    if separate:
        return outs, axes, pose.socket
    out = outs["body"]
    for n in names[1:]:
        np.maximum(out, outs[n], out=out)
    del outs
    # soften the joints between pieces so they read as one sculpt
    blurred = ndimage.gaussian_filter(out, params.get("joint_blur", 1.1) / pitch)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
    w = np.zeros(out.shape, np.float32)
    for c, r in pose.joints:
        d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        w = np.maximum(w, np.clip(1.5 - d / r, 0, 1))
    out = out * (1 - w) + blurred * w
    return out, axes, pose.socket


def to_mesh(field, axes, level=0.5):
    from skimage.measure import marching_cubes
    xs, ys, zs = axes
    pitch = float(xs[1] - xs[0])
    f = np.pad(field, 1, constant_values=0.0)
    v, faces, _, _ = marching_cubes(f, level, spacing=(pitch,) * 3)
    v += np.array([xs[0], ys[0], zs[0]]) - pitch
    m = trimesh.Trimesh(v, faces[:, ::-1], process=True)
    parts = m.split(only_watertight=False)
    m = max(parts, key=lambda q: len(q.faces))          # drop floating scraps
    if m.volume < 0:
        m.invert()
    return m


def load_source(path):
    d = np.load(path)
    s = Source.__new__(Source)
    s.field, s.lo, s.pitch = d["field"], d["lo"], float(d["pitch"])
    return s
