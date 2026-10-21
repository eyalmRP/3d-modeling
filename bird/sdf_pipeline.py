"""Smooth re-pose and color split, with everything as signed fields (negative inside).

Each piece (body, head, tail, neck, cheek tufts) is the exact signed distance of the
source sculpt, sampled through that piece's re-pose transform and clipped to the
piece's region. Pieces are joined with smooth unions, so joints are filleted. Colors
are smooth fields too, so every border is a clean curve; each color's solid is the
part of the figure where that color's field wins.
"""
import numpy as np
from matplotlib.path import Path
from scipy import ndimage

import paint as PT
import stl_repose as R

NAMES = PT.NAMES


# ------------------------------------------------------------------ helpers

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def rounded_slab(u, z, half, bottom, rc):
    """Signed distance to the region |u| < half, z > bottom, with its bottom edges rounded by rc."""
    qx = np.abs(u) - (half - rc)
    qz = (bottom + rc) - z
    out = np.hypot(np.maximum(qx, 0.0), np.maximum(qz, 0.0))
    return out + np.minimum(np.maximum(qx, qz), 0.0) - rc


def poly_sdf(pts, poly):
    """Signed distance (negative inside) from 2D points to a closed polygon."""
    poly = np.asarray(poly, float)
    a, b = poly, np.roll(poly, -1, axis=0)
    d = np.full(len(pts), np.inf)
    for (ax, ay), (bx, by) in zip(a, b):
        ex, ey = bx - ax, by - ay
        t = np.clip(((pts[:, 0] - ax) * ex + (pts[:, 1] - ay) * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(pts[:, 0] - ax - t * ex, pts[:, 1] - ay - t * ey))
    inside = Path(poly).contains_points(pts)
    return np.where(inside, -d, d)


def partition(entries):
    """entries: [(color, field or None)] from highest priority to lowest; None = the rest.
    Returns {color: field} whose negative regions tile space."""
    out, higher = {}, None
    for color, f in entries:
        if f is None:
            g = -higher
        else:
            g = f if higher is None else np.maximum(f, -higher)
            higher = f if higher is None else np.minimum(higher, f)
        out[color] = g if color not in out else np.minimum(out[color], g)
    return out


# ------------------------------------------------------------------ regions (source frame)

def r_head(S):
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    dist = np.linalg.norm(S - R.HEAD_C, axis=-1)
    raised_wing = np.maximum(np.maximum(x + 18.0, z - 70.0), 27.0 - dist)
    return np.maximum(np.maximum(59.0 - z, y - 12.0), -raised_wing)


def r_tail(S):
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    d = (S - R.TAIL_ROOT) @ R.unit(R.TAIL_AXIS)
    r = np.maximum(np.maximum(-d, 22.0 - z), 14.0 - y)
    return np.maximum(r, np.minimum(np.abs(x) - 8.0, 12.0 - d))


def r_scraps(S):
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    zf, yf = np.array(R.CHEST_FRONT).T
    front = np.interp(z, zf, yf)
    return np.maximum(np.maximum(x - 12.0, 26.0 - z), np.maximum(z - 60.0, y - (front - 0.5)))


# ------------------------------------------------------------------ paint (piece frames)

def head_local(S):
    f, l, u = R.head_frame()
    d = S - R.HEAD_C
    return np.stack([d @ l, -(d @ f), d @ u], -1)


def paint_head(H, cfg):
    x, y, z = H[..., 0], H[..., 1], H[..., 2]
    side = np.where(x >= 0, 1.0, -1.0)
    ec = np.where(side[..., None] > 0, PT.EYES[1][0], PT.EYES[-1][0])
    er = np.where(side > 0, PT.EYES[1][1], PT.EYES[-1][1])
    rad = H - ec
    dist = np.linalg.norm(rad, axis=-1) + 1e-9
    e = dist - er
    ax = np.abs(x)
    grow = cfg.get("eye_grow", 0.4)                   # white spreads past the eyeball below the lid
    lid_grow = cfg.get("lid_grow", grow)                # the lid covers the eyeball (and this much more)
    back = (y - ec[..., 1]) - cfg.get("eye_back", 0.35) * er
    n_lid = R.unit(cfg["lid_normal"])
    above = cfg["lid_level"] * er - rad @ n_lid         # negative above the lid line
    lidcap = np.maximum(np.maximum(np.maximum(e - lid_grow, 3.0 - side * x), back), above)
    eye = np.maximum(np.maximum(np.maximum(e - grow, 3.0 - side * x), back), -above)
    g = np.stack([side * cfg["gaze"][0], np.full_like(x, cfg["gaze"][1]), np.full_like(x, cfg["gaze"][2])], -1)
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    cosang = np.einsum("...i,...i->...", rad, g) / dist
    pupil = np.maximum(eye, (np.cos(np.radians(cfg["pupil_deg"])) - cosang) * er)
    beak = np.minimum(y + 4.0, np.maximum(np.maximum(y - 1.0, 13.0 - z), ax - 5.0))
    jaw = np.maximum(np.maximum(np.maximum(-12.0 - z, z + 1.5), np.maximum(y - 6.0, 7.0 - ax)), 1.0 - e)
    q = np.sqrt(((y + 9.0) / 14.0) ** 2 + ((z - 10.5) / 8.5) ** 2)
    yellow = np.maximum(np.maximum(beak, 2.8 - ax), np.maximum(1.5 - z, (q - 1.0) * 8.0))
    throat = np.maximum(np.maximum(z + 9.0, y + 2.0), ax - 10.0)
    lids = np.maximum(np.maximum(e - lid_grow - cfg.get("lid_ring", 2.0),
                                 (y - ec[..., 1]) - cfg.get("lid_back", 2.0)),
                      2.0 - side * x)
    rr = np.linalg.norm(H, axis=-1)
    tuft = np.maximum(np.maximum(16.0 - y, 20.5 - rr), np.maximum(ax - 12.0, z - 6.0))
    bz = cfg["brow_z"]
    brows = np.maximum(np.maximum(np.maximum(bz - z, 4.0 - ax), np.maximum(ax - 23.0, -7.0 - y)),
                       np.maximum(y - 19.0, cfg["brow_gap"] - e))
    brow_cap = np.maximum(np.maximum(e - cfg["brow_cap_depth"], 3.0 - side * x),
                          (ec[..., 2] + cfg["brow_cap_z"]) - z)          # black brow arc over each eye
    brow_cap = np.maximum(brow_cap, (y - ec[..., 1]) - cfg["brow_cap_back"])
    return partition([("black", brow_cap), ("blue", lidcap), ("black", pupil), ("white", eye), ("yellow", yellow),
                      ("red", beak), ("red", jaw), ("black", brows), ("purple", tuft),
                      ("blue", lids), ("white", throat), ("purple", None)])


def paint_body_target(S, P, cfg):
    """Body colors laid out in the re-posed (target) side view: wing with a scalloped white band
    between the purple coverts and the navy primaries, white chest and belly, purple back."""
    shp = P.shape[:-1]
    ax = np.abs(P[..., 0])
    yz = np.stack([P[..., 1].ravel(), P[..., 2].ravel()], 1)
    legs = S[..., 2] - cfg["leg_top"]
    # the wing wraps further forward on the outer side of the body, so it shows from the front too
    xf, kf = cfg.get("wing_wrap", (99.0, 0.0))
    yz_w = yz + np.stack([(kf * np.maximum(ax - xf, 0.0)).ravel(), np.zeros(ax.size)], 1)
    wing = np.maximum(poly_sdf(yz_w, cfg["wing_t"]).reshape(shp), cfg["wing_x"] - ax)
    (ya, za), (yb, zb), hw, amp, period = cfg["band_t"]
    t = np.array([yb - ya, zb - za]) / np.hypot(yb - ya, zb - za)
    n_dir = np.array([-t[1], t[0]])
    dy, dz = P[..., 1] - ya, P[..., 2] - za
    s_ = dy * t[0] + dz * t[1]
    n = dy * n_dir[0] + dz * n_dir[1]                     # > 0 towards the coverts
    upper = hw - amp * np.abs(np.sin(np.pi * s_ / period))
    lower = -hw - amp * np.abs(np.sin(np.pi * (s_ / period + 0.5)))
    navy = np.maximum(wing, n - lower)
    band = np.maximum(wing, np.maximum(lower - n, n - upper))
    chest = poly_sdf(yz, cfg["chest_t"]).reshape(shp)
    return partition([("orange", legs), ("navy", navy), ("white", band), ("purple", wing),
                      ("white", chest), ("purple", None)])


def paint_body(S, cfg, P=None):
    if "wing_t" in cfg:
        return paint_body_target(S, P, cfg)
    x, y, z = np.abs(S[..., 0]), S[..., 1], S[..., 2]
    shp = x.shape
    yz = np.stack([y.ravel(), z.ravel()], 1)
    legs = z - cfg["leg_top"]
    wing = np.maximum(poly_sdf(yz, PT.WING).reshape(shp) - cfg["wing_grow"], 6.0 - x)
    A, B = PT.WING_A, PT.WING_B
    L = np.linalg.norm(B - A)
    axv = (B - A) / L
    t = ((y - A[0]) * axv[0] + (z - A[1]) * axv[1])                 # mm along the wing
    across = -(y - A[0]) * axv[1] + (z - A[1]) * axv[0]
    t1 = (cfg["wing_navy"] + 0.035 * np.abs(np.cos(across * np.pi / 4.5))) * L
    t0 = cfg["wing_white"] * L
    navy = np.maximum(wing, t1 - t)
    band = np.maximum(wing, np.maximum(t0 - t, t - t1))
    white_side = poly_sdf(yz, PT.WHITE_SIDE).reshape(shp)
    white_front = np.maximum(poly_sdf(yz, PT.WHITE_FRONT).reshape(shp), x - 9.0)
    return partition([("orange", legs), ("navy", navy), ("white", band), ("purple", wing),
                      ("white", white_side), ("white", white_front), ("purple", None)])


def paint_tail(S, cfg):
    t = (S - R.TAIL_ROOT) @ R.unit(R.TAIL_AXIS)
    c1, c2 = cfg["tail_white"] * 128.0, cfg["tail_navy"] * 128.0
    return partition([("navy", c2 - t), ("white", c1 - t), ("purple", None)])


def paint_neck(P, pose):
    """White throat in front, purple nape behind, split along the neck's own axis."""
    best = np.full(P.shape[:-1], np.inf)
    side = np.zeros(P.shape[:-1])
    for a, b in zip(pose.neck_pts[:-1], pose.neck_pts[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        Q = a + t[..., None] * ab
        tan = ab / np.linalg.norm(ab)
        back = np.array([0.0, 1.0, 0.0]) - tan[1] * tan
        back /= np.linalg.norm(back)
        d = np.linalg.norm(P - Q, axis=-1)
        closer = d < best
        best = np.where(closer, d, best)
        side = np.where(closer, (P - Q) @ back, side)
    w = side - pose.p.get("nape_offset", 1.5)
    return partition([("white", w), ("purple", None)])


PAINT = {
    "lid_normal": (0.0, 0.22, 1.0), "lid_level": 0.22, "gaze": (0.60, -0.70, 0.20), "pupil_deg": 19.0,
    "brow_z": 12.5, "brow_gap": 2.4, "brow_cap_z": 7.8, "brow_cap_depth": 3.0, "brow_cap_back": 4.0,
    "leg_top": 15.5, "wing_white": 0.37, "wing_navy": 0.44, "wing_grow": 2.0,
    "tail_white": 0.60, "tail_navy": 0.70,
}


# ------------------------------------------------------------------ cheek tufts (new geometry)

def tufts_sdf(H, cfg):
    """Curled feather fans on both sides of the head, behind the eyes (head-local).
    cfg: {"flatten": f, "blend": k, "feathers": [[[x, y, z, r], ...], ...]} - each feather is a
    polyline of points with radii; the fan is thinned front-to-back by `flatten`."""
    if not cfg:
        return np.full(H.shape[:-1], 10.0)
    y0 = cfg["feathers"][0][0][1]
    Q = np.stack([np.abs(H[..., 0]), (H[..., 1] - y0) * cfg["flatten"] + y0, H[..., 2]], -1)
    d = np.full(H.shape[:-1], 10.0)
    for pts in cfg["feathers"]:
        for p0, p1 in zip(pts[:-1], pts[1:]):
            d = smin(d, R.capsule_sdf(Q, p0[:3], p1[:3], p0[3], p1[3]), cfg["blend"])
    return d / cfg["flatten"]


# ------------------------------------------------------------------ legs (new geometry)

def shins_sdf(P, cfg):
    """Thin shins (tapered capsules) from the ankles up into the belly, both sides."""
    d = np.full(P.shape[:-1], 10.0)
    for s in (1.0, -1.0):
        m = np.array([s, 1.0, 1.0])
        d = np.minimum(d, R.capsule_sdf(P, np.multiply(cfg["ankle"], m), np.multiply(cfg["hip"], m),
                                        *cfg["radius"]))
    return d


def cut_for_shins(body, P, cfg):
    """Remove the sculpt's short, thick thighs (keep the feet) and round off the belly bottom."""
    y, z = P[..., 1], P[..., 2]
    slab = np.maximum(np.maximum(cfg["feet_top"] - z, z - cfg["belly"]), y - cfg["slab_back"])
    body = smax(body, -slab, cfg.get("belly_soft", 2.5))
    if "bevel" in cfg:                                     # round off the front of the belly bottom
        (ya, za), (yb, zb) = cfg["bevel"]
        nrm = np.array([zb - za, -(yb - ya)]) / np.hypot(yb - ya, zb - za)     # points front-down
        bev = (y - ya) * nrm[0] + (z - za) * nrm[1]
        body = smax(body, np.minimum(bev, z - (cfg["feet_top"] + 1.0)), 3.0)
    if "rump_cut" in cfg:                                  # nothing behind the line of the wing tips
        (ya, za), (yb, zb), ztop = cfg["rump_cut"]
        nrm = np.array([za - zb, yb - ya]) / np.hypot(yb - ya, zb - za)       # points back-up
        cut = (y - ya) * nrm[0] + (z - za) * nrm[1]
        body = smax(body, np.minimum(np.minimum(cut, ztop - z), z - (cfg["feet_top"] + 1.0)), 2.0)
    keep = rounded_slab(P[..., 0], z, cfg["belly_half"], cfg["belly"], cfg["belly_rc"])
    feet = z - (cfg["feet_top"] + 1.0)                     # leave the feet alone
    return np.maximum(body, np.minimum(keep, feet))


# ------------------------------------------------------------------ assembly

def evaluate(src, pose, P, paint_cfg, tuft_cfg):
    """Union field and per-color fields at points P [..., 3]."""
    # body
    Sb = pose.body_src(P)
    body = src(Sb)
    body = np.maximum(body, Sb[..., 2] - pose.p.get("body_top", 57.0))
    body = np.maximum(body, -r_tail(Sb))
    body = np.maximum(body, -r_scraps(Sb))
    body = np.maximum(body, -np.maximum(Sb[..., 2] - 11.0, np.abs(Sb[..., 0]) - 6.0))
    # head (+ cheek tufts in head-local coordinates)
    Sh = pose.head_src(P)
    head = np.maximum(src(Sh), r_head(Sh))
    Hh = head_local(Sh)
    crest = pose.p.get("crest_trim")                  # trim the sculpt's crest off the back of the head
    if crest:
        rr = np.linalg.norm(Hh, axis=-1)
        head = smax(head, -np.maximum(crest[0] - Hh[..., 1], crest[1] - rr), 1.0)
    head = head * pose.sh
    tuft = tufts_sdf(Hh, tuft_cfg) * pose.sh
    # tail
    St = pose.tail_src(P)
    tail = np.maximum(src(St), r_tail(St)) * float(np.mean(pose.p["tail_scale"]))
    # neck
    neck = pose.neck_sdf(P)
    # new shins
    shins = pose.p.get("shins")
    if shins:
        body = cut_for_shins(body, P, shins)
        leg = shins_sdf(P, shins)

    U = np.minimum(body, head)
    U = smin(U, neck, 2.5)
    if shins:
        U = smin(U, leg, shins["blend"])
    U = smin(U, tail, 2.0)
    U = smin(U, tuft, 1.0)
    U = np.maximum(U, -P[..., 2])                       # flat bottom on the build plate

    pieces = {"body": body, "head": head, "tail": tail, "neck": neck, "tuft": tuft}
    if shins:
        pieces["leg"] = leg
    names = list(pieces)
    D = np.stack([pieces[n] for n in names])
    paints = {
        "body": paint_body(Sb, paint_cfg, P),
        "head": paint_head(Hh, paint_cfg),
        "tail": paint_tail(St, paint_cfg),
        "neck": paint_neck(P, pose),
        "tuft": {"purple": np.full(P.shape[:-1], -1.0)},
        "leg": {"orange": np.full(P.shape[:-1], -1.0)},
    }
    G = {c: np.full(P.shape[:-1], 1e3, np.float32) for c in NAMES}
    for i, n in enumerate(names):
        others = np.min(np.delete(D, i, axis=0), axis=0)
        own = D[i] - others
        for c, f in paints[n].items():
            G[c] = np.minimum(G[c], np.maximum(own, f))
    return U.astype(np.float32), {c: g.astype(np.float32) for c, g in G.items()}


def assemble(src, params, pitch, bounds, paint_cfg=None, tuft_cfg=None, log=print):
    paint_cfg = paint_cfg or PAINT
    (x0, x1), (y0, y1), (z0, z1) = bounds
    x0, y0, z0 = x0 - 0.37 * pitch, y0 - 0.41 * pitch, z0 - 1.0 - 0.43 * pitch   # keep grid off z=0
    xs = np.arange(x0, x1 + pitch, pitch, dtype=np.float32)
    ys = np.arange(y0, y1 + pitch, pitch, dtype=np.float32)
    zs = np.arange(z0, z1 + pitch, pitch, dtype=np.float32)
    pose = R.Pose(params)
    shape = (len(xs), len(ys), len(zs))
    U = np.empty(shape, np.float32)
    G = {c: np.empty(shape, np.float32) for c in NAMES}
    step = max(1, int(1.5e6 // (len(xs) * len(ys))))
    for k in range(0, len(zs), step):
        X, Y, Z = np.meshgrid(xs, ys, zs[k:k + step], indexing="ij")
        P = np.stack([X, Y, Z], -1).astype(np.float64)
        u, g = evaluate(src, pose, P, paint_cfg, tuft_cfg)
        U[:, :, k:k + step] = u
        for c in NAMES:
            G[c][:, :, k:k + step] = g[c]
    # fill internal cavities, drop loose specks
    lab, n = ndimage.label(U > 0)
    if n > 1:
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                           lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        cav = (lab > 0) & ~np.isin(lab, border)
        U[cav] = -np.abs(U[cav]) - 1e-3
        log(f"filled {n - len(border[border > 0])} internal cavities")
    lab, n = ndimage.label(U < 0)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        drop = (lab > 0) & (lab != sizes.argmax())
        U[drop] = np.abs(U[drop]) + 1e-3
        log(f"dropped {n - 1} loose speck(s)")
    del lab
    for f in [U, *G.values()]:
        f[f == 0] = 1e-6                                    # no exact zeros for marching cubes
    return U, G, (xs, ys, zs), pose
