"""Reconstruct the bird from the reference turnaround sheet.

Geometry comes from the drawing's silhouettes: the body, head and wings are built
slice by slice (width from the front view, depth from the side view), the beak is
swept from its side profile with the width taken from the top view, and colors
are projected back from whichever view faces each surface point.

Units are mm. Z is up, the bird faces -Y, the ground is z = 0.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import ref_analysis as ra

S = 100.0 / (888 - 150)       # mm per sheet pixel (side view: ground row 888, brow top row 150)
GROUND = 888
Y0 = 680                       # side-view column of y = 0

# The views don't quite agree, so the front and top views are warped onto the side view
# through matching landmarks (brow top, lid top, eye, jaw, shoulder, wing tip, leg top,
# ankle, ground for the front; beak tip, beak base, head back, wing tip, tail tip for the top).
F_ROWS = (130, 175, 233, 350, 475, 720, 740, 795, 887)
F_TO_S = (150, 195, 250, 365, 490, 732, 742, 797, 888)
T_ROWS = (45, 205, 345, 610, 690, 992)
T_TO_S = (408, 600, 795, 865, 925, 1132)
TOP_K = 1.2                    # the top view is drawn narrower than the front view


def interp(x, xp, fp):
    """np.interp with linear extrapolation past both ends."""
    xp, fp = np.asarray(xp, float), np.asarray(fp, float)
    y = np.interp(x, xp, fp)
    lo = (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x < xp[0], fp[0] + (x - xp[0]) * lo, y)
    return np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * hi, y)


# ------------------------------------------------------------- sheet <-> model

class Sheet:
    def __init__(self):
        self.img, self.masks, self.cls = ra.analyse()
        self.cx = self._mirror_centre(self.masks["front"], 470, 887, 150, 290)
        self.tx = self._mirror_centre(self.masks["top"], 350, 990, 1220, 1340)
        self.parts = self._part_masks()

    @staticmethod
    def _mirror_centre(mask, r0, r1, c0, c1):
        sub = mask[r0:r1]
        best, arg = -1, None
        for c2 in range(2 * c0, 2 * c1):          # candidate centre = c2 / 2
            cols = np.arange(sub.shape[1])
            mir = c2 - cols
            ok = (mir >= 0) & (mir < sub.shape[1])
            score = (sub[:, cols[ok]] & sub[:, mir[ok]]).sum()
            if score > best:
                best, arg = score, c2 / 2
        return arg

    # coordinate maps (vectorised); return fractional sheet coordinates
    @staticmethod
    def side_rc(y, z):
        return GROUND - z / S, Y0 + y / S

    def front_rc(self, x, z):
        srow = GROUND - z / S
        return interp(srow, F_TO_S, F_ROWS), self.cx + x / S

    def top_rc(self, x, y):
        scol = Y0 + y / S
        return interp(scol, T_TO_S, T_ROWS), self.tx + x / (S * TOP_K)

    def _poly(self, pts):
        im = Image.new("1", (self.img.shape[1], self.img.shape[0]), 0)
        ImageDraw.Draw(im).polygon(pts, fill=1)
        return np.asarray(im, bool)

    def is_cls(self, *names):
        m = np.zeros(self.cls.shape, bool)
        for n in names:
            m |= self.cls == ra.NAMES.index(n)
        return m

    def _part_masks(self):
        H, W = self.cls.shape
        rows = np.arange(H)[:, None] * np.ones((1, W), int)
        cols = np.ones((H, 1), int) * np.arange(W)[None, :]
        side, front, top = self.masks["side"], self.masks["front"], self.masks["top"]
        beakish = self.is_cls("red", "yellow", "pink")
        p = {}

        # ---------------- side view
        p["s_beak"] = (ndimage.binary_fill_holes(beakish & side & (rows < 450)) & (cols < 645)) | \
            (side & (cols < 590) & (rows < 440))
        p["s_tuft"] = side & (cols > 796) & (cols < 875) & (rows > 205) & (rows < 340)
        p["s_wing"] = side & self._poly([
            (650, 490), (680, 492), (703, 497), (720, 510), (743, 527), (770, 553), (803, 587),
            (833, 613), (860, 645), (877, 660), (897, 687), (917, 713), (925, 730), (918, 737),
            (903, 735), (853, 730), (803, 721), (770, 708), (737, 688), (703, 661), (683, 648),
            (657, 621), (633, 594), (621, 560), (619, 530), (623, 507), (635, 494)])
        p["s_tail"] = side & self._poly([
            (840, 640), (862, 600), (950, 420), (1060, 230), (1140, 200), (1145, 520),
            (1000, 640), (905, 672), (880, 668)]) & ~p["s_wing"]
        p["s_core"] = side & self._poly([
            (602, 140), (800, 140), (800, 336), (790, 365), (760, 380), (760, 470), (727, 490),
            (743, 527), (770, 553), (803, 587), (833, 613), (843, 628), (828, 668), (806, 710),
            (796, 748), (560, 748), (515, 600), (520, 425), (600, 386), (597, 300)])

        # ---------------- front view (right half is used for the shape, mirrored)
        legs_f = front & (rows > 728) & self.is_cls("red")
        p["f_beak"] = ndimage.binary_fill_holes(beakish & front & (rows < 470))
        p["f_tuft"] = front & (cols > self.cx + 121) & (rows > 215) & (rows < 305)
        p["f_wing"] = front & self._poly([
            (280, 474), (300, 478), (320, 490), (337, 515), (345, 550), (349, 590), (346, 640),
            (339, 680), (326, 710), (311, 724), (300, 719), (294, 700), (284, 680), (270, 650),
            (264, 620), (262, 580), (263, 540), (266, 505), (272, 485)])
        core = front & ~p["f_tuft"] & ~p["f_wing"] & ~legs_f
        low = rows >= 352
        core[low] &= ~p["f_beak"][low]        # below the head the beak hides the neck
        p["f_core"] = core

        # ---------------- top view
        p["t_all"] = top
        return p

    # ---------------- sampling helpers
    @staticmethod
    def sample(mask, r, c):
        r = np.rint(r).astype(int)
        c = np.rint(c).astype(int)
        ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
        out = np.zeros(np.broadcast(r, c).shape, mask.dtype)
        rr, cc = np.broadcast_arrays(r, c)
        out[ok] = mask[rr[ok], cc[ok]]
        return out

    def row_intervals(self, mask, row, c_from, c_to):
        """Runs of True in mask[row, c_from:c_to] as (start, end) sheet columns."""
        row = int(round(float(row)))
        if row < 0 or row >= mask.shape[0]:
            return []
        seg = mask[row, int(c_from):int(c_to)].astype(np.int8)
        d = np.diff(np.concatenate([[0], seg, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        return [(c_from + a, c_from + b) for a, b in zip(starts, ends)]


# ------------------------------------------------------------------ geometry

def superellipse(X, Y, xc, yc, a, b, n=2.3):
    return (np.abs((X - xc) / a) ** n + np.abs((Y - yc) / b) ** n) <= 1.0


def sliced_part(sheet, fmask, smask, X2, Y2, zs, n=2.3, min_px=2, front_hidden=None):
    """Stack of horizontal superellipse slices: width from the front mask (right half,
    mirrored), depth from the side mask. Returns occupancy [nx, ny, nz]."""
    occ = np.zeros(X2.shape + (len(zs),), bool)
    cx = sheet.cx
    for k, z in enumerate(zs):
        frow, _ = sheet.front_rc(0.0, z)
        srow, _ = sheet.side_rc(0.0, z)
        fints = sheet.row_intervals(fmask, frow, cx, cx + 200)
        sints = sheet.row_intervals(smask, srow, 400, 1145)
        if front_hidden is not None:
            fints = front_hidden(frow, fints)
        if not fints or not sints:
            continue
        sl = np.zeros(X2.shape, bool)
        for fa, fb in fints:
            if fb - fa < min_px:
                continue
            x0, x1 = (fa - cx) * S, (fb - cx) * S
            for sa, sb in sints:
                if sb - sa < min_px:
                    continue
                y0, y1 = (sa - Y0) * S, (sb - Y0) * S
                yc, b = (y0 + y1) / 2, (y1 - y0) / 2
                if x0 <= 0.6:                       # touches the midline: one centred shape
                    sl |= superellipse(X2, Y2, 0.0, yc, x1, b, n)
                else:                               # a pair of side features
                    xc, a = (x0 + x1) / 2, (x1 - x0) / 2
                    sl |= superellipse(np.abs(X2), Y2, xc, yc, a, b, n)
        occ[:, :, k] = sl
    return occ


def body_under_wings(sheet):
    """The wings hide the body's sides in the front view; extend the body under them."""
    wing = sheet.parts["f_wing"]

    def fix(frow, fints):
        w = sheet.row_intervals(wing, frow, sheet.cx, sheet.cx + 200)
        if not w or not fints:
            return fints
        wi, wo = w[0][0], w[-1][1]
        out = []
        for a, b in fints:
            if a <= sheet.cx + 4 and b >= wi - 6:
                b = max(b, wi + 0.4 * (wo - wi))
            out.append((a, b))
        return out
    return fix


def neck_behind_beak(sheet):
    """Below the head the beak hides the neck in the front view: fill from the midline."""
    def fix(frow, fints):
        if round(float(frow)) < 352 or frow > 480:
            return fints
        if not fints:                       # fully hidden: keep the visible neck width
            return [(sheet.cx, sheet.cx + 45)]
        a, b = fints[0]
        return [(sheet.cx, max(b, sheet.cx + 40))] + list(fints[1:])
    return fix


def chain_fix(*fns):
    def fix(frow, fints):
        for f in fns:
            fints = f(frow, fints)
        return fints
    return fix


def beak_part(sheet, X, Y, Z):
    """Side profile, elliptical cross-sections; width from the top view, clipped by the front view."""
    s = sheet.parts["s_beak"]
    srow, scol = sheet.side_rc(Y, Z)
    in_side = sheet.sample(s, srow, scol)
    # per side-view column: vertical extent of the beak
    occ = np.zeros(X.shape, bool)
    ys = Y[0, :, 0]
    zs = Z[0, 0, :]
    top = sheet.parts["t_all"]
    for j, y in enumerate(ys):
        col = Y0 + y / S
        col_i = int(round(col))
        if col_i < 400 or col_i >= 645:
            continue
        rr = np.where(s[:, col_i])[0]
        if len(rr) == 0:
            continue
        z_hi, z_lo = (GROUND - rr.min()) * S, (GROUND - rr.max()) * S
        zc, h = (z_hi + z_lo) / 2, (z_hi - z_lo) / 2 + 1e-3
        trow, _ = sheet.top_rc(0.0, y)
        tints = sheet.row_intervals(top, trow, sheet.tx - 160, sheet.tx + 160)
        if not tints:
            continue
        w = max(abs(tints[0][0] - sheet.tx), abs(tints[-1][1] - sheet.tx)) * S * TOP_K
        w = max(w, 1.4)                     # keep the blade printable near the tip
        xs = X[:, j, :]
        zz = Z[:, j, :]
        occ[:, j, :] = (xs / w) ** 2 + ((zz - zc) / h) ** 2 <= 1.0
    # clip with the front view (mirrored right half) so the shield shape matches
    frow, fcol = sheet.front_rc(np.abs(X), Z)
    occ &= sheet.sample(sheet.parts["f_beak"], frow, fcol) & (np.abs(X) < 10.5)
    occ &= in_side
    return occ


def top_clip(sheet, X, Y):
    trow, tcol = sheet.top_rc(X, Y)
    grown = ndimage.binary_dilation(sheet.parts["t_all"], iterations=3)
    return sheet.sample(grown, trow, tcol)[:, :, None]


def capsule_occ(X, Y, Z, a, b, r1, r2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / (ab @ ab), 0, 1)
    d2 = (X - a[0] - t * ab[0]) ** 2 + (Y - a[1] - t * ab[1]) ** 2 + (Z - a[2] - t * ab[2]) ** 2
    return d2 <= (r1 + (r2 - r1) * t) ** 2


def side_yz(col, row):
    return (col - Y0) * S, (GROUND - row) * S


# legs and feet: a striding pose, as in the side view (the near/+x leg is forward)
LEGS = {+1: dict(x=7.0, y=side_yz(704, 0)[0], ay=side_yz(686, 0)[0]),
        -1: dict(x=-7.0, y=side_yz(760, 0)[0], ay=side_yz(752, 0)[0])}


def legs_feet(X, Y, Z):
    occ = np.zeros(X.shape, bool)
    for sgn, leg in LEGS.items():
        x, y = leg["x"], leg["y"]
        knee = (x, y, side_yz(0, 732)[1])
        ankle = (x, leg["ay"], 5.0)
        occ |= capsule_occ(X, Y, Z, knee, ankle, 2.0, 2.0)
        occ |= capsule_occ(X, Y, Z, ankle, ankle, 3.4, 3.4)
        # three forward toes splayed outward (ends at ~8, 15 and 22 mm from centre in the
        # front view), one back toe
        for ang, length in ((4.0, 19.0), (24.0, 20.0), (50.0, 19.5), (180.0, 10.0)):
            a = np.radians(ang)
            tip = (ankle[0] + sgn * np.sin(a) * length, ankle[1] - np.cos(a) * length, 2.7)
            occ |= capsule_occ(X, Y, Z, (ankle[0], ankle[1], 3.0), tip, 2.9, 2.6)
            occ |= capsule_occ(X, Y, Z, tip, tip, 3.0, 3.0)
    return occ


def feather(X, Y, Z, root, tip, width, thick, roll_axis_x):
    """Paddle-shaped feather: narrow root, widest at ~70%, rounded tip."""
    root, tip = np.asarray(root, float), np.asarray(tip, float)
    ax = tip - root
    L = np.linalg.norm(ax)
    ax /= L
    u = np.cross(ax, [0.0, 0.0, 1.0]) if abs(ax[2]) < 0.99 else np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(u, ax)
    c, s_ = np.cos(roll_axis_x), np.sin(roll_axis_x)
    u, v = u * c + v * s_, v * c - u * s_
    px, py, pz = X - root[0], Y - root[1], Z - root[2]
    t = px * ax[0] + py * ax[1] + pz * ax[2]
    uu = px * u[0] + py * u[1] + pz * u[2]
    vv = px * v[0] + py * v[1] + pz * v[2]
    k = np.clip(t / L, 0, 1)
    w = width * np.where(k < 0.72, 0.55 + 0.45 * np.sin(np.pi / 2 * k / 0.72),
                         np.sqrt(np.clip(1 - ((k - 0.72) / 0.28) ** 2, 0, 1)))
    w = np.maximum(w, 0.6)
    return (t >= 0) & (t <= L) & ((uu / w) ** 2 + (vv / (thick / 2)) ** 2 <= 1.0)


def tail(X, Y, Z):
    ra, ta = side_yz(865, 615), side_yz(1130, 255)
    rb, tb = side_yz(885, 642), side_yz(1118, 362)
    occ = feather(X, Y, Z, (0.0, *ra), (0.0, *ta), 5.0, 6.5, 0.0)
    rc, tc = side_yz(890, 648), side_yz(1022, 485)
    for sgn in (+1, -1):
        occ |= feather(X, Y, Z, (sgn * 1.5, *rb), (sgn * 9.0, *tb), 4.6, 3.6, sgn * np.radians(45))
        occ |= feather(X, Y, Z, (sgn * 2.5, *rc), (sgn * 12.0, *tc), 4.2, 3.4, sgn * np.radians(60))
    return occ


def hull_part(sheet, fname, sname, X, Y, Z):
    frow, fcol = sheet.front_rc(np.abs(X), Z)
    srow, scol = sheet.side_rc(Y, Z)
    return sheet.sample(sheet.parts[fname], frow, fcol) & sheet.sample(sheet.parts[sname], srow, scol)


def build(voxel=0.4):
    sheet = Sheet()
    xs = np.arange(-27.0, 27.0 + voxel, voxel)
    ys = np.arange(-40.0, 64.0 + voxel, voxel)
    zs = np.arange(-1.0, 102.0 + voxel, voxel)
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    parts = {}
    parts["core"] = sliced_part(sheet, sheet.parts["f_core"], sheet.parts["s_core"], X2, Y2, zs,
                                front_hidden=chain_fix(neck_behind_beak(sheet), body_under_wings(sheet)))
    parts["wings"] = sliced_part(sheet, sheet.parts["f_wing"], sheet.parts["s_wing"], X2, Y2, zs, n=2.0)
    tclip = top_clip(sheet, X[:, :, 0], Y[:, :, 0])
    parts["legs"] = legs_feet(X, Y, Z)
    parts["tail"] = tail(X, Y, Z)
    parts["beak"] = beak_part(sheet, X, Y, Z)
    parts["tufts"] = hull_part(sheet, "f_tuft", "s_tuft", X, Y, Z)
    for k in ("core", "wings", "beak", "tufts"):
        parts[k] &= tclip
    for k in parts:
        parts[k] &= Z >= 0
    return sheet, (xs, ys, zs), parts
