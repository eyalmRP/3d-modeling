"""Signed-distance model of the cartoon hornbill, split into solid color regions.

Units are millimetres. Z is up, the bird faces -Y, the ground is z = 0.
Every piece has a priority; a point belongs to the highest-priority piece that
contains it, and inside a piece a paint function picks the color. The result is
one closed solid per color, and the solids tile the bird without overlapping.
"""
import numpy as np
from scipy.interpolate import CubicSpline

COLORS = {
    # name: (display RGB, description)
    "purple": ((0.36, 0.33, 0.64), "head, neck, back, wings, tail base, cheek tufts"),
    "navy": ((0.09, 0.08, 0.30), "wing tips and tail tips"),
    "white": ((0.95, 0.95, 0.93), "chest, throat, thighs, eyes, wing and tail spots"),
    "red": ((0.86, 0.05, 0.07), "beak"),
    "yellow": ((0.98, 0.66, 0.06), "beak base patches"),
    "orange": ((0.80, 0.17, 0.05), "legs and feet"),
    "blue": ((0.50, 0.66, 0.94), "eyelids and eye patches"),
    "black": ((0.03, 0.03, 0.04), "eyebrows and pupils"),
}

BIG = np.float32(1e6)


# ---------------------------------------------------------------- primitives

def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def ellipsoid(x, y, z, c, r):
    px, py, pz = (x - c[0]) / r[0], (y - c[1]) / r[1], (z - c[2]) / r[2]
    k0 = np.sqrt(px * px + py * py + pz * pz)
    k1 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def sphere(x, y, z, c, r):
    return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - r


def round_cone(x, y, z, a, b, r1, r2):
    """Capsule whose radius goes from r1 at a to r2 at b (Inigo Quilez)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pax, pay, paz = x - a[0], y - a[1], z - a[2]
    yy = pax * ba[0] + pay * ba[1] + paz * ba[2]
    zz = yy - l2
    qx, qy, qz = pax * l2 - ba[0] * yy, pay * l2 - ba[1] * yy, paz * l2 - ba[2] * yy
    x2 = qx * qx + qy * qy + qz * qz
    y2 = yy * yy * l2
    z2 = zz * zz * l2
    k = np.sign(rr) * rr * rr * x2
    d_mid = (np.sqrt(x2 * a2 * il2) + yy * rr) * il2 - r1
    d_b = np.sqrt(x2 + z2) * il2 - r2
    d_a = np.sqrt(x2 + y2) * il2 - r1
    return np.where(np.sign(zz) * a2 * z2 > k, d_b, np.where(np.sign(yy) * a2 * y2 < k, d_a, d_mid))


def chain(x, y, z, pts, k=0.0):
    """Union of round cones through (x, y, z, r) control points."""
    d = None
    for p, q in zip(pts[:-1], pts[1:]):
        s = round_cone(x, y, z, p[:3], q[:3], p[3], q[3])
        d = s if d is None else (smin(d, s, k) if k > 0 else np.minimum(d, s))
    return d


def extruded(d2, dx, half, rr):
    """Rounded extrusion of a 2D distance d2 by thickness 2*half along dx."""
    wx = d2 + rr
    wy = np.abs(dx) - half + rr
    return np.minimum(np.maximum(wx, wy), 0.0) + np.sqrt(
        np.maximum(wx, 0.0) ** 2 + np.maximum(wy, 0.0) ** 2) - rr


def norm(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------- proportions

HEAD_C, HEAD_R = (0.0, 5.0, 83.5), (15.0, 17.0, 12.8)
BODY_C, BODY_R = (0.0, -3.0, 38.0), (12.5, 15.0, 18.0)
EYE_X, EYE_Z, EYE_R = 9.5, 87.0, 4.2


def head_front_y(x, z, out=0.0):
    """Y of the head's front surface at (x, z), pushed `out` mm outward."""
    cx, cy, cz = HEAD_C
    rx, ry, rz = HEAD_R
    s = max(1.0 - ((x - cx) / rx) ** 2 - ((z - cz) / rz) ** 2, 0.0)
    return cy - ry * np.sqrt(s) - out


EYE_Y = head_front_y(EYE_X, EYE_Z) + 1.5
LOOK = norm((1.5, -3.9, -0.6))          # both pupils glance to the bird's left


def tail_feathers():
    """(root, unit axis, unit across, unit normal, length, half_width_root, half_width_tip, lift)."""
    root = np.array([0.0, 8.0, 38.0])
    main = norm((0.0, 48.0, 46.0))
    xhat = np.array([1.0, 0.0, 0.0])
    normal = norm(np.cross(xhat, main))     # points up/forward, out of the fan
    out = []
    for ang, length, lift in ((0.0, 66.0, 0.0), (-6.0, 61.0, -2.0), (6.0, 61.0, -2.0)):
        a = np.radians(ang)
        axis = norm(main * np.cos(a) + xhat * np.sin(a))
        across = norm(np.cross(normal, axis))
        roll = np.radians(60.0) * np.sign(ang)       # outer feathers droop: a folded, roof-like tail
        across, nrm = across * np.cos(roll) + normal * np.sin(roll), normal * np.cos(roll) - across * np.sin(roll)
        out.append((root + normal * lift, axis, across, nrm, length, 3.3, 4.7))
    return out


TAIL = tail_feathers()

def beak_stations(n=26):
    """Sweep stations (y, z, r) for the closed beak, from its side-view top and bottom profiles."""
    u = np.linspace(0.0, 1.0, 6)
    y = CubicSpline(u, [-4.0, -11.0, -18.0, -25.0, -31.0, -35.5])
    top = CubicSpline(u, [92.0, 91.2, 88.8, 84.5, 78.5, 69.0])
    bot = CubicSpline(u, [64.0, 62.2, 61.6, 62.2, 63.8, 66.0])
    s = np.linspace(0.0, 1.0, n)
    return [(0.0, float(y(t)), float((top(t) + bot(t)) / 2), float((top(t) - bot(t)) / 2)) for t in s]


BEAK = beak_stations()
MOUTH = CubicSpline([-31.0, -27.0, -20.0, -12.0, -6.0, -3.0], [67.0, 67.2, 67.8, 69.0, 71.0, 73.5])

WING_S = np.array([-2.0, 46.5])   # shoulder (y, z)
WING_E = np.array([32.0, 21.0])   # tip (y, z)


# ------------------------------------------------------------------- pieces

def evaluate(X, Y, Z):
    """Return {color: field}; each field is <= 0 inside that color's solid."""
    AX = np.abs(X)
    f32 = np.float32

    # --- base body: head + neck + body + thighs + tail, smoothly blended
    head = ellipsoid(X, Y, Z, HEAD_C, HEAD_R)
    neck = round_cone(X / 0.82, Y, Z, (0, 3.0, 71.0), (0, -4.0, 53.0), 7.2, 8.6) * 0.82
    body = smin(ellipsoid(X, Y, Z, BODY_C, BODY_R),
                ellipsoid(X, Y, Z, (0.0, -7.0, 46.0), (11.5, 10.5, 11.0)), 4.0)   # chest pushed forward
    thigh = ellipsoid(AX, Y, Z, (6.6, -1.5, 22.5), (5.2, 6.2, 6.0))
    rest = smin(smin(smin(head, neck, 6.0), body, 8.0), thigh, 3.0)

    feathers = []
    for root, axis, across, normal, length, w0, w1 in TAIL:
        px, py, pz = X - root[0], Y - root[1], Z - root[2]
        t = px * axis[0] + py * axis[1] + pz * axis[2]
        u = px * across[0] + py * across[1] + pz * across[2]
        n = px * normal[0] + py * normal[1] + pz * normal[2]
        # 2D paddle outline in the fan plane, extruded through its thickness
        outline = round_cone(0.0, u, t, (0, 0, 0), (0, 0, length - w1), w0, w1)
        feathers.append((extruded(outline, n, 1.6, 0.9), t, u, length))
    tail = feathers[0][0]
    for f in feathers[1:]:
        tail = smin(tail, f[0], 1.0)
    base = smin(rest, tail, 5.0)

    # paint on the body: white chest/throat strip, blue eye patches, purple elsewhere
    neckness = np.clip((Z - 48.0) / 10.0, 0.0, 1.0)
    w_line = 2.5 + 4.0 * np.clip((44.0 - Z) / 10.0, 0.0, 1.0) - AX * (0.15 + 0.75 * neckness)
    white = np.maximum(Y - w_line, Z - 71.0)
    eyepatch = ellipsoid(AX, Y, Z, (EYE_X + 1.0, EYE_Y + 1.0, EYE_Z - 0.6), (7.6, 8.0, 6.3))
    body_paint = {
        "blue": eyepatch,
        "white": np.maximum(white, -eyepatch),
        "purple": np.maximum(-white, -eyepatch),
    }

    # paint on the tail: purple base, white drop, navy tip; nearest feather owns
    tail_paint = {"purple": BIG, "white": BIG, "navy": BIG}
    for i, (sd, t, u, length) in enumerate(feathers):
        others = [f[0] for j, f in enumerate(feathers) if j != i]
        own = sd - np.minimum(others[0], others[1])
        cut = 0.58 * length
        drop = ellipsoid(t, u, 0.0, (cut - 0.5, 0.0, 0.0), (5.2, 3.3, 1.0))
        p = {
            "white": drop,
            "navy": np.maximum(cut - t, -drop),
            "purple": np.maximum(t - cut, -drop),
        }
        for c in tail_paint:
            tail_paint[c] = np.minimum(tail_paint[c], np.maximum(own, p[c]))
    o_tail = tail - rest
    base_paint = {}
    for c in ("purple", "white", "blue", "navy"):
        bp = body_paint.get(c, BIG)
        tp = tail_paint.get(c, BIG)
        base_paint[c] = np.minimum(np.maximum(-o_tail, bp), np.maximum(o_tail, tp))

    # --- wings: a rounded slab that hugs the body side and runs past it to the tip
    sy, sz = WING_S
    ey, ez = WING_E
    outline = round_cone(0.0, Y, Z, (0, sy, sz), (0, ey, ez), 9.5, 2.4)
    cx, cy, cz = BODY_C
    rx, ry, rz = (BODY_R[0] + 1.2, BODY_R[1] + 1.2, BODY_R[2] + 1.2)
    s = np.maximum(1.0 - ((Y - cy) / ry) ** 2 - ((Z - cz) / rz) ** 2, 0.0)
    xc = rx * np.sqrt(s)
    xc = -smin(-xc, -np.float32(5.6), 4.0)            # smooth floor so tips stay apart
    wing = extruded(outline, AX - xc, 1.9, 1.0)
    wing = np.maximum(wing, 4.5 - AX)
    axis = (WING_E - WING_S) / np.linalg.norm(WING_E - WING_S)
    a = (Y - sy) * axis[0] + (Z - sz) * axis[1]          # mm along the wing
    b = -(Y - sy) * axis[1] + (Z - sz) * axis[0]         # mm across the wing
    a1 = 12.0
    a2 = 16.5 + 2.2 * np.abs(np.cos(b * np.pi / 5.5))  # scalloped white row
    wing_paint = {
        "purple": a - a1,
        "white": np.maximum(a1 - a, a - a2),
        "navy": a2 - a,
    }

    # --- beak: big downturned upper bill + lower bill, red with yellow side patches
    sx = 0.47 + 0.11 * np.clip((Y + 20.0) / 14.0, 0.0, 1.0)   # broad between the eyes, slim tip
    beak = chain(X / sx, Y, Z, BEAK) * sx
    # shallow mouth-line groove along both sides
    yc = np.clip(Y, -31.0, -3.0)
    groove = np.maximum(np.abs(Z - MOUTH(yc)) - 0.45, -(beak + 0.7))
    groove = np.maximum(groove, np.maximum(Y + 2.5, -31.0 - Y))
    beak = np.maximum(beak, -groove)
    yellow = ellipsoid(AX, Y, Z, (6.5, -12.5, 84.5), (4.6, 9.5, 6.5))
    beak_paint = {"yellow": yellow, "red": -yellow}

    # --- eyes, pupils, lids
    E = (EYE_X, EYE_Y, EYE_Z)
    eyes = sphere(AX, Y, Z, E, EYE_R)
    # pupils: both glance the same way, measured from the nearer eye's centre
    ex, ey_, ez_ = X - EYE_X * np.sign(X), Y - EYE_Y, Z - EYE_Z
    along = ex * LOOK[0] + ey_ * LOOK[1] + ez_ * LOOK[2]
    perp = np.sqrt(np.maximum(ex * ex + ey_ * ey_ + ez_ * ez_ - along * along, 0.0))
    pupils = np.maximum(np.maximum(eyes, perp - 1.35), (EYE_R - 1.6) - along)
    lid_cap = np.maximum(sphere(AX, Y, Z, E, EYE_R + 0.55), (EYE_Z + 0.7) - Z)
    bag = ellipsoid(AX, Y, Z, (EYE_X + 0.3, EYE_Y + 0.6, EYE_Z - 3.9), (4.4, 3.1, 1.8))

    # --- eyebrows: bushy arcs above each eye with spiky tops
    brow_pts = []
    for bx, bz, r in ((3.0, 91.0, 2.2), (6.0, 92.2, 2.7), (10.0, 92.8, 3.0), (14.0, 92.0, 2.6), (17.0, 90.0, 2.0)):
        brow_pts.append((bx, head_front_y(bx, bz, -0.6), bz, r))
    brows = chain(AX, Y, Z, brow_pts, k=1.5)
    for bx, bz, dx, dz in ((5.0, 93.4, -0.5, 1.0), (9.0, 94.6, -0.1, 1.0), (13.0, 94.2, 0.5, 1.0),
                           (16.5, 92.2, 1.0, 0.5)):
        by = head_front_y(min(bx, 14.5), bz, -0.4)
        tip = (bx + dx * 2.2, by - 0.6, bz + dz * 2.2)
        brows = smin(brows, round_cone(AX, Y, Z, (bx, by, bz), tip, 1.6, 0.85), 1.2)

    # --- cheek tufts: small purple feather fans behind the eyes
    root = (11.5, 12.5, 80.5)
    tufts = BIG
    for tip in ((20.0, 13.0, 84.6), (21.3, 13.8, 80.5), (20.0, 13.4, 76.4)):
        tufts = smin(tufts, round_cone(AX, (Y - 13.0) / 0.6 + 13.0, Z, root, tip, 2.4, 1.35) * 0.6, 0.8)

    # --- legs and feet
    legs = round_cone(AX, Y, Z, (7.4, -1.5, 19.0), (8.6, -1.8, 3.4), 2.2, 2.0)
    ank = np.array([8.6, -1.8, 1.7])
    feet = sphere(AX, Y, Z, ank, 2.8)
    for ang, length in ((-14.0, 12.5), (22.0, 14.0), (56.0, 12.0), (188.0, 5.5)):
        a = np.radians(ang)
        d = np.array([np.sin(a), -np.cos(a), 0.0])
        tip = ank + d * length
        toe = round_cone(AX, Y, Z, ank, tip, 2.2, 1.9)
        if ang < 180:
            toe = smin(toe, sphere(AX, Y, Z, tip, 2.3), 0.8)
        feet = smin(feet, toe, 1.0)
    feet = smin(feet, legs, 1.5)

    # ------------------------------------------------ resolve ownership
    ground = -Z
    pieces = [  # highest priority first
        ("brows", brows, {"black": None}),
        ("lids", lid_cap, {"blue": None}),
        ("pupils", pupils, {"black": None}),
        ("eyes", eyes, {"white": None}),
        ("bags", bag, {"blue": None}),
        ("beak", beak, beak_paint),
        ("tufts", tufts, {"purple": None}),
        ("feet", feet, {"orange": None}),
        ("wings", wing, wing_paint),
        ("base", base, base_paint),
    ]
    out = {c: np.full(X.shape, BIG, f32) for c in COLORS}
    higher = np.full(X.shape, BIG, f32)
    for name, sd, paint in pieces:
        sd = np.maximum(sd, ground)
        own = np.maximum(sd, -higher)
        higher = np.minimum(higher, sd)
        for c, p in paint.items():
            region = own if p is None else np.maximum(own, p)
            out[c] = np.minimum(out[c], region).astype(f32)
    return out


BOUNDS = ((-28.0, 28.0), (-42.0, 70.0), (-1.0, 101.0))
