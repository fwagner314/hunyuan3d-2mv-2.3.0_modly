import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# Register torch's DLL directory so custom_rasterizer_kernel (the locally built
# CUDA extension) can find cudart64_12.dll and other native libs at import time.
try:
    _torch_lib = str(Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "torch" / "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import numpy as np
import trimesh
import torch
from PIL import Image

# Shared utilities from the shape bridge
from hunyuan3d_bridge import (
    report, cleanup_cuda,
    setup_paths, _touch_activity,
)
import hunyuan3d_bridge as _bridge

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


_BRIDGE_STALL_TIMEOUT = 45


def _watchdog_loop():
    while True:
        time.sleep(60)
        if time.time() - _bridge._last_activity > _BRIDGE_STALL_TIMEOUT * 60:
            print(json.dumps({"type": "error", "message": f"No progress for {_BRIDGE_STALL_TIMEOUT} min — aborting"}), flush=True)
            os._exit(1)


def _start_watchdog():
    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()


def _histogram_match_pil(src_img, ref_img, mask=None, strength=1.0):
    """Gently match a generated view to the subject colors in a reference.

    The generated-view mask and the reference subject mask are found
    independently: their silhouettes need not occupy the same pixels. Only
    foreground pixels are changed, so white/gray backdrops cannot wash out the
    texture.
    """
    import cv2 as _cv2

    src = np.array(src_img.convert("RGB")).astype(np.float32)
    if ref_img.size != src_img.size:
        ref_img = ref_img.resize(src_img.size, Image.LANCZOS)
    ref = np.array(ref_img.convert("RGB")).astype(np.float32)
    src_lab = _cv2.cvtColor(src.astype(np.uint8), _cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = _cv2.cvtColor(ref.astype(np.uint8), _cv2.COLOR_RGB2LAB).astype(np.float32)

    if mask is not None:
        fg = (mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask))
        fg = np.asarray(fg).astype(np.float32)
        if fg.ndim == 3:
            fg = fg[:, :, 0]
        if fg.shape != src_lab.shape[:2]:
            fg = _cv2.resize(fg.astype(np.uint8) * 255,
                             (src_lab.shape[1], src_lab.shape[0]),
                             interpolation=_cv2.INTER_NEAREST) > 127
        else:
            fg = fg > 0.5
    else:
        fg = np.ones(src_lab.shape[:2], dtype=bool)

    if int(fg.sum()) < 100:
        return src_img

    try:
        ref_fg = _subject_silhouette(ref.astype(np.uint8))
        ref_coverage = float(ref_fg.mean())
        if not (0.02 < ref_coverage < 0.98) or int(ref_fg.sum()) < 100:
            ref_fg = np.ones(ref_lab.shape[:2], dtype=bool)
    except Exception:
        ref_fg = np.ones(ref_lab.shape[:2], dtype=bool)

    corrected = src_lab.copy()
    for ch in range(3):
        src_vals = src_lab[:, :, ch][fg]
        ref_vals = ref_lab[:, :, ch][ref_fg]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        ref_mean, ref_std = ref_vals.mean(), ref_vals.std() + 1e-6
        candidate = np.clip(
            (src_lab[:, :, ch] - src_mean) * (ref_std / src_std) * strength
            + src_mean + strength * (ref_mean - src_mean),
            0, 255)
        corrected[:, :, ch][fg] = candidate[fg]

    result = _cv2.cvtColor(corrected.astype(np.uint8), _cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)

def _flatten_texture(pil_img, strength=0.8):
    """Remove shadows and highlights from a texture, producing flat albedo.

    Uses frequency separation: a large Gaussian blur captures the low-frequency
    lighting gradient, which is then replaced with the image's median colour.
    High-frequency detail (texture grain, edges, seams) is preserved.

    *strength* (0-1) controls how aggressively lighting is removed:
      0 = no change,  1 = fully flat.
    Returns a PIL RGB image.
    """
    import cv2 as _cv2

    arr = np.array(pil_img.convert("RGB")).astype(np.float32)

    # Large-kernel Gaussian blur to extract the lighting gradient.
    _h, _w = arr.shape[:2]
    _ksize = max(3, int(min(_h, _w) * 0.15) | 1)  # ~15% of shortest edge, odd
    blurred = _cv2.GaussianBlur(arr, (_ksize, _ksize), 0)

    # Median colour of the image (robust to outliers).
    _median = np.median(arr.reshape(-1, 3), axis=0)

    # Build a flat version: replace the blurred (lit) component with a uniform
    # colour, while keeping the ratio of detail to blur.
    ratio = np.clip(arr / (blurred + 1e-6), 0.5, 2.0)
    flat = np.clip(_median * ratio, 0, 255)
    result = np.clip(arr * (1 - strength) + flat * strength, 0, 255)

    return Image.fromarray(result.astype(np.uint8))


def _composite_on_white(img):
    """Return an RGB image with the input composited onto a white background.

    RGBA inputs use their alpha channel — unless alpha is mostly opaque (>95%),
    in which case the image is treated as RGB and the background is segmented
    out (handles MV-Adapter grids with fully-opaque alpha + gray backdrop).
    RGB inputs are passed through unless they look like they have a dark/solid
    background, in which case rembg is used to extract the foreground first.
    """
    if img.mode == "RGBA":
        alpha = np.array(img.getchannel("A"))
        if float((alpha > 250).mean()) > 0.95:
            # Fully-opaque alpha: the original behaviour was to paste onto
            # white using the alpha mask, which for an all-opaque image
            # returns the raw RGB unchanged. Do NOT segment here — segmenting
            # eats shaded mid-tones of the subject that sit near the backdrop
            # colour, which mangled the grid splits. Return raw.
            return img.convert("RGB")
        else:
            white = Image.new("RGB", img.size, (255, 255, 255))
            white.paste(img, mask=img.getchannel("A"))
            return white

    # RGB path: try to remove any existing background so the diffusion model
    # always sees the object on white rather than on the original backdrop.
    rgb = img.convert("RGB")
    # Fast path for SYNTHETIC views (MV-Adapter grid quadrants, folder
    # renders): a flat/vignetted solid backdrop is segmented exactly by the
    # border-median rule — no neural model involved. rembg's u2net returns an
    # all-opaque matte on synthetic cutouts, which silently passed the gray
    # 128 backdrop through; gray-backdrop references are out-of-distribution
    # for the paint model and made it hallucinate black/zebra side views.
    _arr = np.array(rgb)
    # Use corner patches (5x5) for background color — corners are almost always
    # pure background, even when the subject touches the border edges.
    _h, _w = _arr.shape[:2]
    _patch = max(3, min(24, min(_h, _w) // 100))
    _corners = np.concatenate([
        _arr[:_patch, :_patch].reshape(-1, 3),
        _arr[:_patch, -_patch:].reshape(-1, 3),
        _arr[-_patch:, :_patch].reshape(-1, 3),
        _arr[-_patch:, -_patch:].reshape(-1, 3),
    ]).astype(int)
    _corner_std = _corners.std(0).max()
    if _corner_std > 20:
        # Subject touches a corner — can't reliably estimate background
        pass
    else:
        _bg_color = _corners.mean(0)
        _border = np.concatenate([_arr[0, :], _arr[-1, :], _arr[:, 0], _arr[:, -1]]).astype(int)
        _dev = np.abs(_border - _bg_color).max(axis=1)
        _p90 = float(np.percentile(_dev, 90))
        if _p90 <= 25:
            _fg = _subject_silhouette(_arr, _bg_color=_bg_color, _T=max(40.0, min(100.0, 2.0 * _p90 + 15.0)))
            if 0.02 < float(_fg.mean()) < 0.98:
                _out = _arr.copy()
                _out[~_fg] = 255
                return Image.fromarray(_out)
    try:
        from rembg import remove, new_session
        sess = new_session(providers=["CPUExecutionProvider"])
        rgba = remove(rgb, session=sess, bgcolor=[255, 255, 255, 0])
        # If rembg wiped almost everything, fall back to the original image.
        alpha = np.array(rgba.getchannel("A"))
        if alpha.max() < 20 or (alpha > 10).sum() < (rgba.width * rgba.height * 0.01):
            return rgb
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.getchannel("A"))
        return white
    except Exception as _e:
        return rgb


def _subject_silhouette(img_np, tol=None, _bg_color=None, _T=None):
    """Foreground mask of a subject on a uniform (possibly vignetted) backdrop.

    Corner flood-fill (even with FLOODFILL_FIXED_RANGE) breaks on the mild
    radial vignette of model-rendered backgrounds: every corner seed only
    fills pixels near its OWN brightness, so background bands survive inside
    the "subject" mask — which poisoned the masked normalisation stats, the
    outline cleanup, the warp masks and the white-composite fast path
    (references kept their gray backdrop and the paint model hallucinated
    black/zebra sides from the out-of-distribution conditioning).

    New rule: background = pixels within an adaptive colour distance of the
    BORDER MEDIAN that are connected to the image border; subject = the rest.
    The threshold uses the p90 of border deviation, not the median: a photo
    with a shadow touching one edge has a BIMODAL border whose median
    deviation stays tiny while p90 exposes the second colour (that is what
    keeps real photos on the rembg path in _composite_on_white).

    When called from _composite_on_white, _bg_color and _T are provided
    (corner-based background, adaptive threshold) to handle subjects that
    touch the border edges.
    """
    import cv2 as _cv
    arr = img_np.astype(np.int16)
    if _bg_color is not None and _T is not None:
        # Caller provided corner-based bg color and threshold
        med = np.array(_bg_color)
        T = float(_T)
    else:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
        med = np.median(border, axis=0)
        p90 = float(np.percentile(np.abs(border - med).max(axis=1), 90))
        # Wider than the border's own spread: a radial vignette is darkest at the
        # corners (on the border) and brightest at the centre (never sampled), so
        # the threshold must exceed the border p90 to swallow the whole backdrop.
        # Genuinely low-contrast subjects (gray-on-gray) get swallowed too — the
        # coverage guards then fall back gracefully instead of corrupting.
        T = float(tol) if tol else max(30.0, min(90.0, 3.0 * p90 + 20.0))
    dist = np.abs(arr - med).max(axis=2)
    bg = (dist < T).astype(np.uint8)
    _n, lab = _cv.connectedComponents(bg)
    edge_labels = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    if len(edge_labels):
        bg = np.isin(lab, edge_labels).astype(np.uint8)
    fg = (1 - bg).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    fg = _cv.morphologyEx(fg, _cv.MORPH_OPEN, k, iterations=1)
    fg = _cv.morphologyEx(fg, _cv.MORPH_CLOSE, k, iterations=1)
    return fg.astype(bool)


def _remove_silhouette_outline(pil_img, band=None, abs_lum_cap=45):
    """Erase the near-black contour ring that the delight/diffusion models
    draw around the subject silhouette.

    Works at any resolution: the band is sized from the image height (outlines
    thicken proportionally when 512px generations are upscaled to the 4096px
    render/atlas resolution).  Candidates are dark pixels in the outer band of
    the subject that sit directly against a BRIGHT interior: a model-drawn
    outline is always followed by the object's real surface colour, while
    genuinely dark object parts (e.g. wooden legs, tires, dark trim) stay
    dark deeper inside and are therefore left untouched.  A whole-dark or
    whole-light subject is skipped entirely (relative thresholds + coverage
    guards), so unusual subjects can never be eaten away.
    """
    import cv2 as _cv
    arr = np.array(pil_img.convert("RGB"))
    h = arr.shape[0]
    if band is None:
        band = max(3, min(48, int(round(5 * h / 512))))
    fg = _subject_silhouette(arr).astype(np.uint8)
    if not (0.02 < float(fg.mean()) < 0.98):
        return pil_img
    k3 = np.ones((3, 3), np.uint8)
    band1 = fg - _cv.erode(fg, k3, iterations=band)            # outer ring
    band2 = _cv.erode(fg, k3, iterations=band) - \
            _cv.erode(fg, k3, iterations=band + 5)             # just inside it
    if band2.sum() == 0:
        return pil_img
    lab = _cv.cvtColor(arr, _cv.COLOR_RGB2LAB)
    l_med = float(np.median(lab[:, :, 0][fg.astype(bool)]))
    thr = min(abs_lum_cap, 0.22 * l_med)
    dark = (lab[:, :, 0].astype(np.float32) < thr).astype(np.uint8)
    bright = (lab[:, :, 0].astype(np.float32) > 0.55 * l_med).astype(np.uint8)
    # Dilate the bright-interior signal far enough inward-to-outward to cover
    # the whole outer band (band+2 iterations ≈ band+2 px of reach).
    bright_inside = _cv.dilate(band2 * bright, k3, iterations=band + 2)
    m = (band1 * dark * bright_inside).astype(bool)
    if m.sum() < 10:
        return pil_img
    m_u8 = _cv.dilate(m.astype(np.uint8) * 255, k3, iterations=1)
    out = _cv.inpaint(arr, m_u8, max(3, band // 2), _cv.INPAINT_TELEA)
    return Image.fromarray(out)


def _mesh_silhouette_from_pos(pos_img):
    """Derive the mesh silhouette from a position map image.

    Position maps render the background as white (255,255,255); mesh pixels
    carry 3D position colours. Returns a bool mask True where the mesh is.
    """
    arr = np.asarray(pos_img.convert("RGB")).astype(np.int16)
    return (np.abs(arr - 255).max(axis=-1)) > 8


def _silhouette_contour(sil_mask, n=128):
    """Outer contour of a silhouette, resampled to `n` points by arc length."""
    import cv2 as _cv
    sil_u8 = (sil_mask * 255).astype(np.uint8)
    contours, _ = _cv.findContours(sil_u8, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=_cv.contourArea).reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return None
    targets = np.linspace(0, total, n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(c) - 1)
    frac = np.zeros(n)
    valid = idx < len(c) - 1
    frac[valid] = (targets[valid] - cum[idx[valid]]) / np.maximum(d[np.clip(idx[valid], 0, len(d) - 1)], 1e-9)
    nxt = c[np.clip(idx + 1, 0, len(c) - 1)]
    return c[idx] + frac[:, None] * (nxt - c[idx])


def _polar_contour(sil_mask, n=128):
    """Silhouette boundary resampled at uniform POLAR angle around the area
    centroid.  This is the correspondence anchor for the TPS warp: arc-length
    anchoring keys point #0 to wherever cv2 happened to start tracing, so two
    silhouettes of the same object (photo cutout vs mesh render) routinely end
    up matching the chair top against a leg — the interior then collapses into
    the smeared blobs we saw in the bake.  Polar anchoring is stable for any
    upright camera pair; non star-shaped spans fall back to max radius."""
    import cv2 as _cv
    sil_u8 = (sil_mask.astype(np.uint8)) * 255
    contours, _ = _cv.findContours(sil_u8, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=_cv.contourArea).reshape(-1, 2).astype(np.float64)
    ys, xs = np.where(sil_mask)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    ang = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
    radius = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    edges = np.linspace(-np.pi, np.pi, n + 1)
    bins = np.clip(np.digitize(ang, edges) - 1, 0, n - 1)
    r_b = np.zeros(n)
    np.maximum.at(r_b, bins, radius)
    cnt = np.bincount(bins, minlength=n)
    has = cnt > 0
    if not has.any():
        return None
    if not has.all():
        filled = np.where(has)[0]
        for i in np.where(~has)[0]:
            d = np.abs(filled - i)
            r_b[i] = r_b[filled[d.argmin()]]
    a_c = (edges[:-1] + edges[1:]) / 2
    return np.stack([cx + r_b * np.cos(a_c), cy + r_b * np.sin(a_c)], axis=1)


def _deform_view_to_silhouette(ref_img, pos_img, out_size):
    """Warp a reference view so its subject silhouette matches the mesh's.

    Two candidate warps are computed and the better one wins, scored by the
    IoU between the warped subject and the mesh silhouette:

      1. bbox-similarity warp: scale+translate the subject bbox onto the mesh
         bbox.  Rigid (no local bending) so it can never melt — the floor.
      2. TPS on polar-angle contour correspondence: smooth bending that also
         fixes shape differences.  The old arc-length correspondence keyed
         point 0 to cv2's arbitrary contour start pixel, which mapped top to
         side and collapsed the interior into smeared blobs; polar anchoring
         is start-pixel independent.

    Output is the warped subject composited on white, masked to the mesh
    silhouette.  Raises ValueError when even the best warp matches poorly
    (<0.30 IoU) so callers can fall back to diffusion/raw views instead of
    baking garbage.
    """
    import cv2 as _cv
    from skimage.transform import ThinPlateSplineTransform, warp as _skwarp

    ref_pil = ref_img.convert("RGB").resize((out_size, out_size), Image.LANCZOS)
    ref_np = np.asarray(ref_pil).astype(np.float32)
    ref_u8 = np.asarray(ref_pil)

    mesh_sil = _mesh_silhouette_from_pos(pos_img)
    if (mesh_sil.shape[0], mesh_sil.shape[1]) != (out_size, out_size):
        _u8 = Image.fromarray((mesh_sil * 255).astype(np.uint8)).resize(
            (out_size, out_size), Image.NEAREST)
        mesh_sil = np.asarray(_u8) > 0
    if not mesh_sil.any():
        return ref_pil  # no mesh visible from this camera — nothing to align to

    ref_sil = _subject_silhouette(ref_u8)
    if not ref_sil.any():
        return ref_pil

    def _iou(cand_u8):
        cs = _subject_silhouette(cand_u8)
        inter = np.logical_and(cs, mesh_sil).sum()
        union = np.logical_or(cs, mesh_sil).sum()
        return float(inter) / max(float(union), 1.0)

    # ── floor: bbox similarity warp (dst->src inverse affine) ─────────────
    ry, rx = np.where(ref_sil)
    my, mx = np.where(mesh_sil)
    r_cx, r_cy = (rx.min() + rx.max()) / 2.0, (ry.min() + ry.max()) / 2.0
    m_cx, m_cy = (mx.min() + mx.max()) / 2.0, (my.min() + my.max()) / 2.0
    sx = (rx.max() - rx.min() + 1) / max(mx.max() - mx.min() + 1, 1)
    sy = (ry.max() - ry.min() + 1) / max(my.max() - my.min() + 1, 1)
    # Scale sanity: the renderer fits the mesh to frame and MV-Adapter frames
    # its subjects to 90%, so a legit reference-vs-mesh area ratio is ~1.
    # Anything beyond ~4x one way means the "mesh silhouette" is a speck or
    # covers the whole frame (broken position map) — no warp can fix that,
    # and IoU is blind to it since scaling can force silhouettes to agree.
    if not (0.22 <= (sx * sy) <= 4.5):
        raise ValueError(f"mesh/reference area scale insane (sx={sx:.2f}, sy={sy:.2f})")
    M_inv = np.float32([[sx, 0, r_cx - sx * m_cx], [0, sy, r_cy - sy * m_cy]])
    best = _cv.warpAffine(ref_u8, M_inv, (out_size, out_size),
                          flags=_cv.INTER_LINEAR, borderMode=_cv.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    best_iou = _iou(best)
    how = f"bbox(iou={best_iou:.2f})"

    # ── refinement: TPS on polar correspondence ────────────────────────────
    ref_pts = _polar_contour(ref_sil)
    mesh_pts = _polar_contour(mesh_sil)
    if ref_pts is not None and mesh_pts is not None:
        try:
            try:
                tps = ThinPlateSplineTransform.from_estimate(src=mesh_pts, dst=ref_pts)
            except AttributeError:  # skimage < 0.26
                tps = ThinPlateSplineTransform()
                tps.estimate(src=mesh_pts, dst=ref_pts)
            w = _skwarp(ref_np, inverse_map=tps, output_shape=(out_size, out_size),
                        order=1, mode="constant", cval=255.0, clip=True)
            w = np.clip(w, 0, 255).astype(np.uint8)
            iou_t = _iou(w)
            if iou_t > best_iou + 0.02:
                best, best_iou, how = w, iou_t, f"tps-polar(iou={iou_t:.2f})"
        except Exception:
            pass

    if best_iou < 0.30:
        raise ValueError(f"silhouette match too poor after warp ({how})")

    out = np.full((out_size, out_size, 3), 255, np.uint8)
    out[mesh_sil] = best[mesh_sil]
    # Interior holes: mesh-silhouette regions the reference view does not
    # cover (thinner/offset legs, extra arm area) would otherwise keep the
    # flat backdrop colour and bake as gray patches or "metallic" legs.
    # Inpaint them from the surrounding real subject pixels.
    try:
        _wsub = _subject_silhouette(out)
        _holes = (mesh_sil & ~_wsub).astype(np.uint8)
        if 0 < int(_holes.sum()) < int(0.35 * mesh_sil.sum()):
            _holes = _cv.dilate(_holes, np.ones((3, 3), np.uint8), iterations=1)
            out = _cv.inpaint(out, _holes * 255, max(3, out_size // 128),
                              _cv.INPAINT_TELEA)
    except Exception:
        pass
    print(json.dumps({"type": "log", "message":
        f"[warp] mesh-silhouette alignment: {how}"}), flush=True)
    return Image.fromarray(out, "RGB")


def _sil_match_score(ref_img, pos_img, out_size):
    """IoU between a reference's subject silhouette (bbox-aligned onto the
    mesh silhouette) and the mesh's rendered silhouette. Cheap pre-warp test
    used to detect mirrored left/right conventions between view generators
    (MV-Adapter names left/right opposite to hunyuan's camera order)."""
    mesh_sil = _mesh_silhouette_from_pos(pos_img)
    if (mesh_sil.shape[0], mesh_sil.shape[1]) != (out_size, out_size):
        _u8 = Image.fromarray((mesh_sil * 255).astype(np.uint8)).resize(
            (out_size, out_size), Image.NEAREST)
        mesh_sil = np.asarray(_u8) > 0
    ref_sil = _subject_silhouette(np.array(ref_img.convert("RGB").resize(
        (out_size, out_size), Image.LANCZOS)))
    if not mesh_sil.any() or not ref_sil.any():
        return 0.0
    ry, rx = np.where(ref_sil)
    my, mx = np.where(mesh_sil)
    m_h, m_w = my.max() - my.min() + 1, mx.max() - mx.min() + 1
    sub = ref_sil[ry.min():ry.max() + 1, rx.min():rx.max() + 1]
    sub_r = np.asarray(Image.fromarray((sub * 255).astype(np.uint8)).resize(
        (int(m_w), int(m_h)), Image.NEAREST)) > 127
    canvas = np.zeros((out_size, out_size), bool)
    canvas[my.min():my.min() + sub_r.shape[0], mx.min():mx.min() + sub_r.shape[1]] = sub_r
    inter = float(np.logical_and(canvas, mesh_sil).sum())
    union = float(np.logical_or(canvas, mesh_sil).sum())
    return inter / max(union, 1.0)


def _guided_blend(warped_np, diff_np, mesh_sil):
    """Per-pixel blend between a warped reference and the diffusion view using
    LAB-similarity as a confidence map.  Where the two agree (same fabric
    color, aligned detail) the reference's real pixels are kept; where they
    disagree (misaligned edges, background bleed, projection offset) the
    diffusion view's correct geometry wins.  Smooth Gaussian feathering at
    silhouette boundaries eliminates any hard transition artefacts.  A final
    edge cleanup pass replaces any residual background-colored pixels near
    the silhouette boundary with the nearest subject color."""
    import cv2 as _cv2_gb
    warped_lab = _cv2_gb.cvtColor(warped_np, _cv2_gb.COLOR_RGB2LAB).astype(np.float32)
    diff_lab = _cv2_gb.cvtColor(diff_np, _cv2_gb.COLOR_RGB2LAB).astype(np.float32)
    dist = np.sqrt(np.sum((warped_lab - diff_lab) ** 2, axis=2))
    conf = np.exp(-dist / 30.0)
    conf = np.where(mesh_sil, conf, 0.0).astype(np.float32)
    _h, _w = conf.shape
    _k = max(3, min(_h, _w) // 128 * 3) | 1
    conf = _cv2_gb.GaussianBlur(conf, (_k, _k), 0)
    conf = np.clip(conf, 0.0, 1.0)[..., None]
    blended = warped_np.astype(np.float32) * conf + diff_np.astype(np.float32) * (1.0 - conf)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    # Edge bleed cleanup: replace background-colored pixels near the
    # silhouette boundary with nearest subject color via inpaint.
    try:
        _bsil = _subject_silhouette(blended).astype(np.uint8)
        _bring = _cv2_gb.dilate(_bsil, np.ones((3, 3), np.uint8), iterations=3)
        _bring = (_bring - _bsil).astype(np.uint8) * 255
        if _bring.sum() > 0:
            blended = _cv2_gb.inpaint(blended, _bring, 3, _cv2_gb.INPAINT_TELEA)
    except Exception:
        pass
    return blended


def _hybrid_warp_multiview(ref_views, diff_views, position_maps, out_size):
    """Per-view hybrid assembly: views backed by a real reference carry the
    full-resolution ORIGINAL pixels warped onto the mesh silhouette with the
    thin-plate-spline; unreferenced views keep the diffusion output.

    Orientation auto-pick: for the two side cameras, both candidate side
    references are scored against this camera's mesh silhouette and the
    better fit is warped — this absorbs mirrored left/right naming between
    view generators and hunyuan's camera order instead of shredding the warp.

    Strict acceptance: a warp is only used when its subject actually covers
    the mesh silhouette (coverage >= 0.90 and IoU >= 0.60). Rejected warps
    fall back to the diffusion view for that camera, logged."""
    out = []
    n_ref = len(ref_views)
    for i, diff_view in enumerate(diff_views):
        _touch_activity()
        fallback = diff_view.convert("RGB").resize((out_size, out_size), Image.LANCZOS)
        if i < n_ref:
            ref_idx = i
            if i in (1, 3) and n_ref >= 4:
                alt = 3 if i == 1 else 1
                s_own = _sil_match_score(ref_views[i], position_maps[i], out_size)
                s_alt = _sil_match_score(ref_views[alt], position_maps[i], out_size)
                if s_alt > s_own + 0.05:
                    ref_idx = alt
                    print(json.dumps({"type": "log", "message":
                        f"[hybrid] cam{i}: mirrored convention detected — using ref#{alt} "
                        f"(sil iou {s_own:.2f} vs {s_alt:.2f})"}), flush=True)
            try:
                warped = _deform_view_to_silhouette(
                    ref_views[ref_idx], position_maps[i], out_size)
                wa = np.array(warped)
                mesh_sil = _mesh_silhouette_from_pos(position_maps[i])
                if (mesh_sil.shape[0], mesh_sil.shape[1]) != (out_size, out_size):
                    mesh_sil = np.asarray(Image.fromarray(
                        (mesh_sil * 255).astype(np.uint8)).resize(
                        (out_size, out_size), Image.NEAREST)) > 0
                wsub = _subject_silhouette(wa)
                cov = float(np.logical_and(mesh_sil, wsub).sum()) / max(float(mesh_sil.sum()), 1.0)
                iou = float(np.logical_and(wsub, mesh_sil).sum()) / max(
                    float(np.logical_or(wsub, mesh_sil).sum()), 1.0)
                if cov >= 0.85 and iou >= 0.50:
                    # Warp passes gate — but still blend with diffusion
                    # for alignment safety: the warped reference brings real
                    # fabric color/detail, the diffusion view brings correct
                    # geometry; the guided blend keeps the best of both.
                    try:
                        blended = _guided_blend(
                            np.array(warped), np.array(fallback), mesh_sil)
                        out.append(Image.fromarray(blended))
                    except Exception:
                        out.append(warped)
                    continue
                print(json.dumps({"type": "log", "message":
                    f"[hybrid] view {i} warp rejected (cov={cov:.2f}, iou={iou:.2f}) "
                    f"— using diffusion view"}), flush=True)
            except Exception as _e:
                print(json.dumps({"type": "log", "message":
                    f"[hybrid] view {i} warp failed ({_e}), using diffusion view"}), flush=True)
        out.append(fallback)
    return out


def _split_tiled_image(image_path, count=4):
    """Split a tiled image into a grid based on the image's aspect ratio.

    Uses closest-distance matching against candidate grids (same logic as
    the shape bridge's _detect_grid) so both pipelines agree on layouts.

    Returns cropped PIL images composited onto white, up to `count`.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception:
        return [image_path]
    w, h = img.size
    ratio = w / h if h > 0 else 1.0
    _candidates = [(2, 1), (2, 2), (2, 3)]
    _best, _best_err = (2, 2), float("inf")
    for _nc, _nr in _candidates:
        _err = abs(ratio - _nc / _nr)
        if _err < _best_err:
            _best, _best_err = (_nc, _nr), _err
    _ncols, _nrows = _best
    _cw, _ch = w // _ncols, h // _nrows
    if _cw < 8 or _ch < 8:
        return [_composite_on_white(img)]
    _views = []
    for idx in range(min(count, _ncols * _nrows)):
        _ri = idx // _ncols
        _ci = idx % _ncols
        _cell = img.crop((_ci * _cw, _ri * _ch, (_ci + 1) * _cw, (_ri + 1) * _ch))
        _views.append(_cell.convert("RGB"))
    return _views


def _resolve_source_image(args):
    """Find the single image the user wired in for texturing."""
    source = args.get("image_path") or ""
    return source if source and os.path.exists(source) else ""


def _select_reference_views(args, count, input_mode="tiled"):
    """Pick `count` reference images for the diffusion model.

    count == 1: the whole wired image is used as a single reference (no split).
    count >= 2: the wired image is split into a grid (auto-detected from aspect
                ratio) and the first `count` views are returned.

    When input_mode is "single", always returns the whole image as one reference
    regardless of `count`.

    Returns a list of PIL images (length == count), or [] if nothing resolved.
    """
    source = _resolve_source_image(args)
    if not source:
        return []
    img = Image.open(source).convert("RGBA")
    img = _composite_on_white(img)

    if input_mode == "single":
        return [img]

    if count <= 1:
        return [img]

    quads = _split_tiled_image(source, count)
    if len(quads) < 2:
        return [img]
    return quads[:count]


def _bridge_first_load(args):
    """First-load bridge: link weights Modly placed anywhere (sibling node
    dirs / HF cache) into the layout hy3dgen expects, and repair out-of-extension
    files (venv site-packages) if needed. Non-fatal — logs and moves on."""
    try:
        from hunyuan3d_bootstrap import ensure_bridged
        model_dir = args.get("model_dir") or args.get("model_cache") or ""
        siblings = []
        if model_dir and os.path.isdir(os.path.dirname(model_dir)):
            _parent = os.path.dirname(model_dir)
            siblings = [os.path.join(_parent, d) for d in os.listdir(_parent)
                        if os.path.isdir(os.path.join(_parent, d)) and d != os.path.basename(model_dir)]
        ensure_bridged({
            "ext_dir": str(Path(__file__).resolve().parent),
            "model_dir": model_dir,
            "node_id": "texture",
            "siblings": siblings,
        })
    except Exception as e:
        print(json.dumps({"type": "log",
            "message": f"[bridge] first-load bootstrap skipped: {e}"}), flush=True)


def texture_mesh(args):
    """Texture an existing mesh with the Hunyuan3D-2.0 paint pipeline.

    Stages (each reported with status + subtext + percentage):
      load mesh -> decimate -> load paint models -> UV-unwrap ->
      render normal/position multiviews -> delight conditioning image ->
      multiview diffusion -> bake textures -> inpaint -> export GLB.
    """
    BRIDGE_BUILD = "2026-10-04-chart-fill-color-preserve"
    print(json.dumps({"type": "log", "message":
        f"[texture] bridge build {BRIDGE_BUILD}"}), flush=True)
    _bridge_first_load(args)

    mesh_path = args.get("mesh_path", "")
    if not mesh_path or not os.path.exists(mesh_path):
        print(json.dumps({"type": "error", "message": f"mesh_path not found: {mesh_path}"}), flush=True)
        return

    output_path = args.get("output_path", "output.glb")
    DECIMATE_FACES = int(args.get("decimate_faces", 40000))
    TEXTURE_SIZE = int(args.get("texture_size", 2048))
    TEXTURE_DIFFUSION_STEPS = int(args.get("texture_diffusion_steps", 30) or 30)
    if TEXTURE_DIFFUSION_STEPS < 1:
        TEXTURE_DIFFUSION_STEPS = 1
    if TEXTURE_DIFFUSION_STEPS > 100:
        TEXTURE_DIFFUSION_STEPS = 100

    # Delight (lighting normalization) is OFF by default: it re-renders the
    # subject under canonical lighting and often shifts/washes out the real
    # colors and detail. Pass delight="on" to re-enable the stock behaviour.
    delight = str(args.get("delight", "off") or "off").lower()

    # Flatten texture: remove all shadows/highlights from the baked texture
    # to produce a flat albedo suitable for PBR lighting.
    # Flatten texture: RETIRED — normalization + delight handle lighting.
    flatten_texture = False

    # Texture generation method.
    #   diffusion: Hunyuan3D-2 multiview diffusion (512px ceiling).
    #   hybrid:    diffusion for unreferenced views + full-res originals
    #              warped onto the mesh silhouettes for referenced views.
    #   deform:    RETIRED — aliased to hybrid (same warp, plus diffusion
    #              fills top/bottom instead of leaving them to inpaint).
    texture_method = str(args.get("texture_method", "diffusion") or "diffusion").lower()
    if texture_method == "deform":
        print(json.dumps({"type": "log", "message":
            "[texture] 'deform' retired — running 'hybrid' (same warp + diffusion fill)"}), flush=True)
        texture_method = "hybrid"
    if texture_method not in ("diffusion", "hybrid"):
        texture_method = "diffusion"

    # How many reference images we are feeding the diffusion model. 1 = whole
    # image as a single reference; 2/3/4 = take that many quadrants from the
    # 2x2 tile in reading order [front, left, back, right].
    input_mode = str(args.get("input_mode", "tiled")).lower()
    reference_images = int(args.get("reference_images", 4) or 4)
    if input_mode == "single":
        reference_images = 1
    if reference_images < 1:
        reference_images = 1
    if reference_images > 6:
        reference_images = 6

    cond_views = _select_reference_views(args, reference_images, input_mode)
    if not cond_views:
        print(json.dumps({"type": "error", "message": "Conditioning image not found"}), flush=True)
        return

    print(json.dumps({"type": "log", "message": f"Using {len(cond_views)} reference image(s) for texturing"}), flush=True)

    # Capture the source colour histogram BEFORE any processing so we can
    # correct colour drift in later stages (delight, diffusion, bake).
    _source_histogram_ref = cond_views[0].convert("RGB").copy()
    # Untouched full-resolution copies of the reference views. Hybrid mode
    # warps THESE (not the delight/diffusion re-renders) onto the mesh
    # geometry, so the baked texture carries the real original pixels.
    _raw_cond_views = [v.convert("RGB").copy() for v in cond_views]

    # Equal diffusion-view weights preserve the model's saturated colors.
    # Hybrid mode prioritizes directly supplied reference views over generated
    # views where their UV projections overlap.
    if texture_method == "hybrid":
        _stock_grad = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
        _dynamic_weights = [1.0 if _i < len(cond_views) else _stock_grad[_i]
                            for _i in range(6)]
    else:
        _dynamic_weights = [1.0] * 6
    _view_weights_override = _dynamic_weights
    print(json.dumps({"type": "log", "message": f"[texture] dynamic view weights: {_view_weights_override}"}), flush=True)

    _start_watchdog()

    report(3, "Loading input mesh", os.path.basename(mesh_path))
    mesh = trimesh.load(mesh_path)
    if mesh is None:
        print(json.dumps({"type": "error", "message": "Failed to load mesh"}), flush=True)
        return
    # trimesh may return a Scene (GLB container); extract the single geometry.
    if isinstance(mesh, trimesh.Scene):
        _geoms = [g for g in mesh.geometry.values() if hasattr(g, "faces")]
        mesh = _geoms[0] if _geoms else None
    if mesh is None or not hasattr(mesh, "faces"):
        print(json.dumps({"type": "error", "message": "Mesh has no geometry"}), flush=True)
        return
    report(8, "Mesh loaded", f"{len(mesh.vertices)} verts / {len(mesh.faces)} faces")

    # Pre-UV-unwrap simplification (cheap face reduction before the paint pass).
    if DECIMATE_FACES > 0 and hasattr(mesh, "faces") and len(mesh.faces) > DECIMATE_FACES:
        report(12, "Decimating mesh", f"target ~{DECIMATE_FACES} faces")
        try:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=int(DECIMATE_FACES))
            except TypeError:
                mesh = mesh.simplify_quadric_decimation(target_count=int(DECIMATE_FACES))
            report(18, "Decimation done", f"{len(mesh.faces)} faces")
        except Exception as _dec_err:
            print(json.dumps({"type": "log", "message": f"[warn] decimation skipped: {_dec_err}"}), flush=True)

    # Mesh sanity: flipped or inconsistent face winding makes back_project's
    # cos map go negative, so those faces bake as HOLES no matter how many
    # views cover them — and on the mesh they reappear as the obvious
    # "doesn't blend" inpainted patches (inner arms, seat tops). Repair
    # winding/normals in memory (vertex positions untouched) and log stats.
    try:
        _wind_before = bool(getattr(mesh, "is_winding_consistent", False))
        _watertight = bool(getattr(mesh, "is_watertight", False))
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        print(json.dumps({"type": "log", "message":
            f"[texture] mesh sanity: winding_consistent_before={_wind_before} "
            f"watertight={_watertight} -> winding/normals repaired"}), flush=True)
    except Exception as _nr_err:
        print(json.dumps({"type": "log", "message":
            f"[texture] mesh sanity skipped: {_nr_err}"}), flush=True)

    report(22, "Loading Hunyuan3D-2.0 paint models", "delight + multiview diffusion")
    # hy3dgen_path points at the Hunyuan3D-2 (2.0) folder so `hy3dgen.texgen`
    # resolves to the 2.0 paint pipeline.
    hy3dgen_path = args.get("hy3dgen_path", "")
    if hy3dgen_path and os.path.isdir(hy3dgen_path):
        sys.path.insert(0, hy3dgen_path)

    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
    import hy3dgen.texgen.utils.uv_warp_utils as _uvwarp_module

    # hy3dgen's stock mesh_uv_wrap(mesh) only takes the mesh and leaves every
    # xatlas option at its default — that produces hundreds of tiny single-face
    # charts and a sparse atlas full of black gaps, which the inpaint stage
    # then has to bridge (blurry). It also lacks the atlas_size / max_cost /
    # uv_stats params the texture pipeline relies on. Replace it with the
    # full-API version (patch lives here, not in site-packages, so it survives
    # venv rebuilds).
    def _mesh_uv_wrap_patched(mesh, atlas_size=0, padding=2, max_cost=8.0,
                              max_iterations=3, brute_force=False,
                              rotate_charts=True, bilinear=True):
        import trimesh as _trimesh
        import xatlas as _xatlas
        if isinstance(mesh, _trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if len(mesh.faces) > 500000000:
            raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")
        _atlas = _xatlas.Atlas()
        _atlas.add_mesh(mesh.vertices.astype('float32'), mesh.faces.astype('uint32'))
        _chart_opts = _xatlas.ChartOptions()
        _chart_opts.max_cost = max_cost
        _chart_opts.max_iterations = max_iterations
        _chart_opts.fix_winding = True
        _pack_opts = _xatlas.PackOptions()
        _pack_opts.resolution = atlas_size
        _pack_opts.padding = padding
        _pack_opts.bruteForce = brute_force
        _pack_opts.rotate_charts = rotate_charts
        _pack_opts.bilinear = bilinear
        _atlas.generate(_chart_opts, _pack_opts, verbose=False)
        vmapping, indices, uvs = _atlas.get_mesh(0)
        mesh.vertices = mesh.vertices[vmapping]
        mesh.faces = indices
        mesh.visual.uv = uvs
        try:
            mesh.metadata['uv_stats'] = {
                'charts': _atlas.chart_count,
                'width': _atlas.width,
                'height': _atlas.height,
                'utilization': float(_atlas.utilization),
            }
        except Exception:
            pass
        return mesh

    _uvwarp_module.mesh_uv_wrap = _mesh_uv_wrap_patched
    mesh_uv_wrap = _mesh_uv_wrap_patched

    # Stock hy3dgen's Multiview_Diffusion_Net.__call__ hardcodes
    # num_inference_steps=30 and accepts no such kwarg, so the user's
    # "Texture Diffusion Steps" parameter would be ignored (and the call
    # below would crash). Restore the tunable signature (patch lives here, not
    # in site-packages, so it survives venv rebuilds).
    from hy3dgen.texgen.utils import multiview_utils as _mv_module

    def _mv_call_patched(self, input_images, control_images, camera_info,
                         num_inference_steps=30):
        from typing import List as _List
        self.seed_everything(0)
        if not isinstance(input_images, _List):
            input_images = [input_images]
        input_images = [input_image.resize((self.view_size, self.view_size))
                        for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))
        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]
        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image
        mvd_image = self.pipeline(input_images, num_inference_steps=num_inference_steps, **kwargs).images
        return mvd_image

    _mv_module.Multiview_Diffusion_Net.__call__ = _mv_call_patched

    # diffusers' encode_prompt returns the negative embeddings on CPU when CFG
    # is disabled (default in hy3dgen for non-turbo), while the positive
    # embeddings come back on cuda — the very next torch.cat then raises
    # "Expected all tensors to be on the same device". The old patched
    # site-packages had an explicit device sync here; stock hy3dgen removed it.
    # Re-apply it (patch lives here, not in site-packages, so it survives venv
    # rebuilds).
    import diffusers as _diffusers_cls
    _orig_encode_prompt = _diffusers_cls.StableDiffusionPipeline.encode_prompt

    def _encode_prompt_synced(self, *args, **kwargs):
        _out = _orig_encode_prompt(self, *args, **kwargs)
        if isinstance(_out, tuple) and len(_out) >= 2:
            _pe, _npe = _out[0], _out[1]
            if torch.is_tensor(_pe) and torch.is_tensor(_npe) and _npe.device != _pe.device:
                _out = (_pe, _npe.to(_pe.device)) + tuple(_out[2:])
        return _out

    _diffusers_cls.StableDiffusionPipeline.encode_prompt = _encode_prompt_synced

    # Stock hy3dgen tuned the delight stage down (cfg_image 2.5 -> 1.5, steps
    # 75 -> 50). Restore the original values so delight=on produces the same
    # results as the old patched site-packages (patch lives here, not in
    # site-packages, so it survives venv rebuilds).
    import hy3dgen.texgen.utils.dehighlight_utils as _delight_module

    _orig_delight_init = _delight_module.Light_Shadow_Remover.__init__

    def _delight_init_patched(self, config):
        _orig_delight_init(self, config)
        self.cfg_image = 2.5

    _delight_module.Light_Shadow_Remover.__init__ = _delight_init_patched

    def _delight_call_patched(self, image):
        import numpy as _np
        import cv2 as _cv2
        # Save the original at 512 so we can restore its edge pixels after
        # the delight model's edge artifacts (halos, color shifts, outline).
        _orig_512 = image.convert("RGB").resize((512, 512))
        image = image.resize((512, 512))
        if image.mode == 'RGBA':
            image_array = _np.array(image)
            alpha_channel = image_array[:, :, 3]
            erosion_size = 3
            kernel = _np.ones((erosion_size, erosion_size), _np.uint8)
            alpha_channel = _cv2.erode(alpha_channel, kernel, iterations=1)
            image_array[alpha_channel == 0, :3] = 255
            image_array[:, :, 3] = alpha_channel
            image = Image.fromarray(image_array)
        image = image.convert('RGB')
        image = self.pipeline(
            prompt="",
            image=image,
            generator=torch.manual_seed(42),
            height=512,
            width=512,
            num_inference_steps=75,
            image_guidance_scale=self.cfg_image,
            guidance_scale=self.cfg_text,
        ).images[0]
        # Match the delit image's colour distribution back to the original using
        # LAB-space histogram matching — preserves original colours while keeping
        # the shadow/highlight removal from the delight step.  The match is
        # restricted to the subject so the white background doesn't dilute it,
        # and the delight model's signature black silhouette outline is erased
        # before it can propagate into every diffusion view and the bake.
        #
        # IMPORTANT: the anchor is the FRONT reference's subject palette, NOT
        # this view's own original.  Matching each side view back to its own
        # original re-imposed that view's directional lighting on top of the
        # delight pass (dark MV-Adapter side views came back dark), so the
        # diffusion model then painted near-black sides that no normalisation
        # step can recover texture from.  A gentle strength here only keeps
        # the global palette honest; per-view lighting equalisation happens
        # later in the luminance-balance step.
        image = _remove_silhouette_outline(image)
        # Restore original edges: the delight model + outline removal create
        # halos and colour shifts at subject boundaries ("separation").  Blend
        # the original's edge pixels back using a feathered subject mask so
        # interior keeps delight's lighting while edges preserve original detail.
        try:
            _orig_mask = _subject_silhouette(np.array(_orig_512))
            if float(_orig_mask.mean()) > 0.02:
                _k = max(5, min(15, 512 // 50)) | 1
                _mask_f = _cv2.GaussianBlur(
                    _orig_mask.astype(np.float32), (_k, _k), 0)[..., None]
                _del_np = np.array(image).astype(np.float32)
                _orig_np = np.array(_orig_512).astype(np.float32)
                image = Image.fromarray(np.clip(
                    _del_np * _mask_f + _orig_np * (1.0 - _mask_f),
                    0, 255).astype(np.uint8))
        except Exception:
            pass
        try:
            _dm = _subject_silhouette(np.array(image))
            if not (0.02 < float(_dm.mean()) < 0.98):
                _dm = None
        except Exception:
            _dm = None
        image = _histogram_match_pil(image, _source_histogram_ref, mask=_dm, strength=0.5)
        return image

    _delight_module.Light_Shadow_Remover.__call__ = _delight_call_patched

    # diffusers >= 0.39 refuses to execute the custom 'hunyuanpaint' pipeline
    # code (bundled with hy3dgen itself) without trust_remote_code=True, but
    # hy3dgen's Multiview_Diffusion_Net does not pass it. Inject the flag for
    # this process so the paint pipeline loads. The code being executed is the
    # installed hy3dgen dependency — trusted local code, not remote.
    import diffusers as _diffusers
    _orig_from_pretrained = _diffusers.DiffusionPipeline.from_pretrained.__func__

    def _from_pretrained_trust_remote(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return _orig_from_pretrained(cls, *args, **kwargs)

    _diffusers.DiffusionPipeline.from_pretrained = classmethod(_from_pretrained_trust_remote)

    # Patch the multiview pipeline's RGBA->RGB conversion so transparent
    # backgrounds are composited onto WHITE instead of the stock gray (127,127,127).
    # White matches the training distribution of the multiview model and prevents
    # dark backgrounds from pulling the generated views down.
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline as _HPP_cls
    from hy3dgen.texgen.hunyuanpaint import pipeline as _hunyuanpaint_pipeline
    import numpy as _np
    from PIL import Image as _Image
    def _to_rgb_white(maybe_rgba):
        if maybe_rgba.mode == 'RGB':
            return maybe_rgba
        elif maybe_rgba.mode == 'RGBA':
            rgba = maybe_rgba
            white = _np.full((rgba.size[1], rgba.size[0], 3), 255, dtype=_np.uint8)
            white = _Image.fromarray(white, 'RGB')
            white.paste(rgba, mask=rgba.getchannel('A'))
            return white
        else:
            raise ValueError("Unsupported image type.", maybe_rgba.mode)
    _hunyuanpaint_pipeline.to_rgb_image = _to_rgb_white

    # When delight is off (default), skip loading the ~1.5 GB Light_Shadow_Remover
    # entirely. The stock load_models() loads it unconditionally; we patch the
    # class so only the multiview model is built, and make cpu-offload skip the
    # missing delight model. This meaningfully lowers the VRAM/RAM ceiling on
    # 6 GB cards where the delight model is never used anyway.
    if delight != "on":
        from hy3dgen.texgen.utils.multiview_utils import Multiview_Diffusion_Net as _MVNet

        def _load_models_no_delight(self):
            torch.cuda.empty_cache()
            self.models['multiview_model'] = _MVNet(self.config)

        def _offload_no_delight(self, gpu_id=None, device="cuda"):
            self.models['multiview_model'].pipeline.enable_model_cpu_offload(
                gpu_id=gpu_id, device=device)

        _HPP_cls.load_models = _load_models_no_delight
        _HPP_cls.enable_model_cpu_offload = _offload_no_delight
        print(json.dumps({"type": "log", "message": "[texture] delight off — skipping delight model load (~1.5 GB saved)"}), flush=True)

    paint_model = args.get("paint_model", "")
    paint_subfolder = "hunyuan3d-paint-v2-0"  # always standard; turbo is broken
    model_cache = args.get("model_cache", "")
    # Resolve the snapshot root locally (ignore any bad path passed in). Search
    # model_cache and its sibling node dirs for the paint subfolder.
    _candidates = [model_cache] if model_cache else []
    if model_cache:
        _parent = os.path.dirname(model_cache)
        if os.path.isdir(_parent):
            _candidates += [os.path.join(_parent, d) for d in os.listdir(_parent)
                            if os.path.isdir(os.path.join(_parent, d)) and d != os.path.basename(model_cache)]
    for _c in _candidates:
        if _c and os.path.isfile(os.path.join(_c, paint_subfolder, "model_index.json")):
            paint_model = _c
            break
    if not paint_model:
        print(json.dumps({"type": "error", "message": f"Paint weights not found under {model_cache} or siblings."}), flush=True)
        return
    # Use the STANDARD (non-turbo) paint model. The turbo variant's 2p5D UNet
    # skips the `ref_scale_timing` assignment inside its reference-attention
    # block, so the reference images are effectively ignored and every output
    # view collapses to one averaged color. The standard model applies
    # ref_scale_timing = ref_scale and uses CFG (ref_scale=[0,1]), which is
    # what actually conditions the texture on the input views. This matches the
    # known-good oldmodel config (texture_variant="hunyuan3d-paint-v2-0").
    _saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        # from_pretrained(model_path, subfolder=...) expects model_path to be
        # the snapshot root and resolves delight + multiview as siblings of the
        # subfolder. If paint_model is already the snapshot root, pass subfolder.
        _snap_root = str(paint_model)
        if os.path.isfile(os.path.join(_snap_root, paint_subfolder, "model_index.json")):
            print(json.dumps({"type": "log", "message": f"[texture] loading paint model subfolder: {paint_subfolder}"}), flush=True)
            pipeline = Hunyuan3DPaintPipeline.from_pretrained(_snap_root, subfolder=paint_subfolder)
        else:
            # from_pretrained defaults subfolder='hunyuan3d-paint-v2-0-turbo' which
            # constructs a broken multiview path. Always pass the standard subfolder.
            pipeline = Hunyuan3DPaintPipeline.from_pretrained(paint_model, subfolder=paint_subfolder)
    finally:
        torch.set_default_dtype(_saved_dtype)

    # Apply dynamic view weights based on how many reference images exist.
    if '_view_weights_override' in locals():
        pipeline.config.candidate_view_weights = _view_weights_override
        print(json.dumps({"type": "log", "message": f"[texture] applied view weights: {pipeline.config.candidate_view_weights}"}), flush=True)

    try:
        pipeline.enable_model_cpu_offload()
    except Exception as _offload_err:
        print(json.dumps({"type": "log", "message": f"[warn] cpu offload skipped: {_offload_err}"}), flush=True)

    # Low-VRAM throughput optimizations (6 GB): attention slicing and VAE
    # slicing cap peak memory per forward pass so the run doesn't spill into
    # slow Windows shared GPU memory, which is what makes later steps drag.
    try:
        pipeline.enable_attention_slicing()
    except Exception as _attn_err:
        print(json.dumps({"type": "log", "message": f"[warn] attention slicing skipped: {_attn_err}"}), flush=True)
    try:
        pipeline.enable_vae_slicing()
    except Exception as _vae_err:
        print(json.dumps({"type": "log", "message": f"[warn] vae slicing skipped: {_vae_err}"}), flush=True)

    # The bake rasterizes each generated view to (re)project it onto the UVs.
    # back_project produces only RENDER_RES^2 points scattered onto the
    # TEXTURE_SIZE^2 atlas. If RENDER_RES is too small, the points are too sparse
    # -> empty bake mask -> black mesh. Keep at least a 512-pixel raster buffer so
    # the mesh always projects enough pixels, even when the requested atlas is tiny.
    # Final atlas is still TEXTURE_SIZE (resized at the end if needed).
    RENDER_RES = min(max(TEXTURE_SIZE, 512), 4096)
    pipeline.config.render_size = RENDER_RES
    pipeline.config.texture_size = TEXTURE_SIZE
    pipeline.render.set_default_render_resolution(RENDER_RES)
    pipeline.render.set_default_texture_resolution(TEXTURE_SIZE)

    # bake_exp 4 suppressed grazing-angle faces (inner arms, seat rims) into
    # holes that inpaint then fills with detail-free patches. 2 keeps their
    # real (stretched but textured) texels; frontal views still win the
    # cos-weighted merge. Hole fraction is logged after bake so this stays
    # measurable/revertable.
    pipeline.config.bake_exp = 2
    pipeline.render.bake_angle_thres = 85
    pipeline.render.bake_unreliable_kernel_size = 2

    import cv2 as _cv2
    def _chart_inpaint(texture, mask, vtx_pos, pos_idx, vtx_uv, uv_idx):
        """Fill UV charts from their own baked pixels and nearby surface color."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        image = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
        height, width, channels = image.shape
        if channels != 3:
            raise ValueError(f"UV inpaint expects an RGB texture, got {image.shape}")
        bake_mask = np.asarray(mask).squeeze()
        if bake_mask.shape != (height, width):
            raise ValueError(
                f"UV mask shape {bake_mask.shape} does not match texture {(height, width)}")
        uvs = np.asarray(vtx_uv, dtype=np.float32)
        faces = np.asarray(uv_idx, dtype=np.int64)
        if uvs.ndim != 2 or uvs.shape[1] != 2:
            raise ValueError(f"UV vertices must have shape (N, 2), got {uvs.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"UV faces must be triangles, got {faces.shape}")
        if faces.size and (faces.min() < 0 or faces.max() >= len(uvs)):
            raise ValueError("UV face indices are outside the UV vertex array")
        if not faces.size or not np.any(bake_mask):
            return image

        px = np.clip(np.rint(uvs[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
        py = np.clip(np.rint((1 - uvs[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
        rows = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
        cols = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(uvs), len(uvs))).tocsr()
        chart_count, labels = connected_components(graph, directed=False)
        face_labels = labels[faces[:, 0]]
        face_order = np.argsort(face_labels, kind="stable")
        face_starts = np.concatenate((
            [0], np.cumsum(np.bincount(face_labels, minlength=chart_count))))
        vertex_order = np.argsort(labels, kind="stable")
        vertex_starts = np.concatenate((
            [0], np.cumsum(np.bincount(labels, minlength=chart_count))))

        # xatlas normally gives matching position and UV indices. If it does
        # not, reconstruct 3D positions from corresponding face corners. UV
        # coordinates remain a safe fallback for other renderer layouts.
        positions = uvs
        mesh_positions = np.asarray(vtx_pos, dtype=np.float32)
        mesh_faces = np.asarray(pos_idx, dtype=np.int64)
        if (len(mesh_positions) == len(uvs) and
                mesh_faces.shape == faces.shape and np.array_equal(mesh_faces, faces)):
            positions = mesh_positions
        elif mesh_faces.shape == faces.shape and mesh_faces.size and (
                mesh_faces.min() >= 0 and mesh_faces.max() < len(mesh_positions)):
            uv_flat = faces.ravel()
            mesh_flat = mesh_faces.ravel()
            counts = np.bincount(uv_flat, minlength=len(uvs)).astype(np.float32)
            positions = np.zeros((len(uvs), mesh_positions.shape[1]), dtype=np.float32)
            for axis in range(mesh_positions.shape[1]):
                positions[:, axis] = np.bincount(
                    uv_flat, weights=mesh_positions[mesh_flat, axis],
                    minlength=len(uvs)) / np.maximum(counts, 1)

        def chart_region(chart_id):
            part = face_order[face_starts[chart_id]:face_starts[chart_id + 1]]
            if part.size == 0:
                return None
            ids = vertex_order[vertex_starts[chart_id]:vertex_starts[chart_id + 1]]
            x0 = max(0, int(px[ids].min()) - 2)
            x1 = min(width, int(px[ids].max()) + 3)
            y0 = max(0, int(py[ids].min()) - 2)
            y1 = min(height, int(py[ids].max()) + 3)
            occupancy = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            for face in faces[part]:
                points = np.column_stack((px[face] - x0, py[face] - y0))
                _cv2.fillConvexPoly(occupancy, points, 255)
            known = (occupancy > 0) & (bake_mask[y0:y1, x0:x1] > 0)
            return ids, x0, x1, y0, y1, occupancy, known

        # A single projected texel is not enough to choose the color of an
        # entire island. Use well-sampled charts as 3D nearest neighbors for
        # islands with zero or one baked pixels.
        support_counts = np.zeros(chart_count, dtype=np.int64)
        support_colors = np.zeros((chart_count, 3), dtype=np.uint8)
        for chart_id in range(chart_count):
            region = chart_region(chart_id)
            if region is None:
                continue
            _, x0, x1, y0, y1, _, known = region
            support_counts[chart_id] = int(known.sum())
            if support_counts[chart_id] >= 2:
                support_colors[chart_id] = np.median(
                    image[y0:y1, x0:x1][known], axis=0).astype(np.uint8)
        support_ids = np.flatnonzero(support_counts[labels] >= 2)
        if not support_ids.size:
            holes = np.where(bake_mask > 0, 0, 255).astype(np.uint8)
            return _cv2.inpaint(image, holes, 3, _cv2.INPAINT_TELEA)
        support_tree = cKDTree(positions[support_ids])

        output = image.copy()
        core_mask = np.zeros((height, width), dtype=np.uint8)
        sparse_charts = 0
        for chart_id in range(chart_count):
            region = chart_region(chart_id)
            if region is None:
                continue
            ids, x0, x1, y0, y1, occupancy, known = region
            occupied = occupancy > 0
            if not occupied.any():
                continue
            chart_image = image[y0:y1, x0:x1].copy()
            if support_counts[chart_id] >= 2:
                chart_image[~known] = 0
                holes = np.where(known, 0, 255).astype(np.uint8)
                if holes.any():
                    chart_image = _cv2.inpaint(chart_image, holes, 3, _cv2.INPAINT_TELEA)
            else:
                sparse_charts += 1
                distances, near_indices = support_tree.query(positions[ids])
                nearest = near_indices[np.argmin(distances)]
                source_chart = labels[support_ids[nearest]]
                chart_image[:] = support_colors[source_chart]
            output[y0:y1, x0:x1][occupied] = chart_image[occupied]
            core_mask[y0:y1, x0:x1][occupied] = 255

        # Preserve all directly baked texels, including thin triangles that
        # rasterize between pixel centers and fall outside the chart mask.
        remaining = np.where((core_mask > 0) | (bake_mask > 0), 0, 255).astype(np.uint8)
        if remaining.any():
            output = _cv2.inpaint(output, remaining, 3, _cv2.INPAINT_TELEA)
        print(json.dumps({"type": "log", "message":
            f"[texture] UV chart fill: {chart_count} components, "
            f"{sparse_charts} sparse charts, "
            f"{float((core_mask > 0).mean()):.1%} chart coverage"}), flush=True)
        return output

    def _patched_uv_inpaint(self, texture, mask):
        if isinstance(texture, np.ndarray):
            tex_np = texture
        elif isinstance(texture, Image.Image):
            tex_np = np.asarray(texture).astype(np.float32) / 255.0
        else:
            tex_np = texture.cpu().numpy()
        if tex_np.dtype == np.uint8 or (tex_np.size and float(np.nanmax(tex_np)) > 1.5):
            tex_np = tex_np.astype(np.float32) / 255.0
        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()
        print(json.dumps({"type": "log", "message":
            f"[texture] UV seam fill: {len(vtx_uv)} UV vertices, "
            f"{len(uv_idx)} UV faces, {len(pos_idx)} position faces"}), flush=True)
        return _chart_inpaint(tex_np, mask, vtx_pos, pos_idx, vtx_uv, uv_idx)

    import types
    pipeline.render.uv_inpaint = types.MethodType(_patched_uv_inpaint, pipeline.render)
    _orig_fast_bake = pipeline.render.fast_bake_texture
    def _patched_fast_bake(self, textures, cos_maps):
        channel = textures[0].shape[-1]
        tex_merge = torch.zeros(self.texture_size + (channel,), device=self.device)
        trust = torch.zeros(self.texture_size + (1,), device=self.device)
        for t, c in zip(textures, cos_maps):
            view_sum = (c > 0).sum()
            if view_sum > 0:
                painted = ((c > 0) * (trust > 0)).sum()
                if painted.float() / view_sum.float() > 0.999:
                    continue
            tex_merge += t * c
            trust += c
        tex_merge = tex_merge / torch.clamp(trust, min=1E-8)
        return tex_merge, trust > 1E-8
    pipeline.render.fast_bake_texture = types.MethodType(_patched_fast_bake, pipeline.render)

    report(32, "Paint models loaded", "starting texture pass")

    import gc

    # Debug: dump split views to output dir for inspection
    _debug_dir = os.path.join(os.path.dirname(output_path) or ".", "debug_textures")
    os.makedirs(_debug_dir, exist_ok=True)
    _view_names = ["front", "left", "back", "right", "top", "bottom"]
    for _i, (_v, _n) in enumerate(zip(cond_views, _view_names)):
        _v.save(os.path.join(_debug_dir, f"01_split_{_i}_{_n}.png"))

    # ── Stage 1: Delight conditioning views (optional) ───────────────────
    # Delight re-renders the subject under canonical lighting; OFF by default
    # because it washes out real colors/detail. When off we keep the raw
    # reference images as-is. Deform mode always skips it (model not loaded).
    if delight == "on" and texture_method != "deform":
        report(40, "Delighting reference views", f"removing shadows/highlights from {len(cond_views)} view(s)")
        # Isolate subjects onto clean white backgrounds before delight so that
        # delight's colour stats and the subsequent masked match are computed
        # on the subject alone — not contaminated by backdrop pixels.
        try:
            from rembg import remove as _rembg_delight
            _cleaned = []
            for _v in cond_views:
                try:
                    _rgba = _rembg_delight(_v.convert("RGB"))
                    _white = Image.new("RGB", _v.size, (255, 255, 255))
                    _white.paste(_rgba, mask=_rgba.getchannel("A"))
                    _cleaned.append(_white)
                except Exception:
                    _cleaned.append(_v.convert("RGB"))
            cond_views = _cleaned
            print(json.dumps({"type": "log", "message":
                f"[texture] rembg-isolated {len(cond_views)} views before delight"}), flush=True)
        except ImportError:
            pass  # rembg not available; proceed without isolation
        cond_views = [pipeline.recenter_image(v.convert("RGB")) for v in cond_views]
        cond_views = [pipeline.models["delight_model"](v) for v in cond_views]
        for _i, (_v, _n) in enumerate(zip(cond_views, _view_names)):
            _v.save(os.path.join(_debug_dir, f"02_delight_{_i}_{_n}.png"))
        del pipeline.models["delight_model"]
        gc.collect()
        torch.cuda.empty_cache()
        report(46, "Delight done", "freed ~1 GB")
    else:
        report(40, "Skipping delight", "using raw reference views")
        # rembg's matte can leave a dark fringe on the cutout edge; erase it
        # so it can't seed the outline artifact in downstream views.
        cond_views = [_remove_silhouette_outline(v) for v in cond_views]
        for _i, (_v, _n) in enumerate(zip(cond_views, _view_names)):
            _v.save(os.path.join(_debug_dir, f"02_delight_{_i}_{_n}.png"))
        report(46, "Delight skipped", "")

    # ── Stage 2: UV-unwrap and load mesh into renderer ───────────────────
    report(48, "UV-unwrapping mesh", "preparing for baking")
    # xatlas unwrap: higher max_cost merges tiny single-face islands into
    # fewer, larger charts so the atlas has less black space to inpaint.
    # max_cost scales with face count — high-poly meshes need much more
    # aggressive merging to avoid hundreds of scattered micro-charts.
    # Raising the upper bound and iteration count directly targets the
    # 38% hole fraction observed with the previous conservative settings.
    _nfaces = len(mesh.faces)
    _mc = max(8.0, min(256.0, _nfaces / 80.0))
    mesh = mesh_uv_wrap(mesh, atlas_size=TEXTURE_SIZE, padding=2,
                        max_cost=_mc, max_iterations=5, rotate_charts=True)
    _uvs = getattr(mesh, "metadata", {}).get("uv_stats")
    if _uvs:
        print(json.dumps({"type": "log", "message":
            f"[texture] UV unwrap: {_uvs['charts']} charts, "
            f"{_uvs['utilization'] * 100:.1f}% packed, {_uvs['width']}x{_uvs['height']}"
            f" (max_cost={_mc:.0f}, {_nfaces} faces)"}),
            flush=True)
    pipeline.render.load_mesh(mesh)

    # ── Stage 3: Render normal/position maps from 6 cameras ──────────────
    elevs = pipeline.config.candidate_camera_elevs
    azims = pipeline.config.candidate_camera_azims
    view_weights = pipeline.config.candidate_view_weights

    report(52, "Rendering normal/position maps", "6 camera views")
    normal_maps = pipeline.render_normal_multiview(elevs, azims, use_abs_coor=True)
    position_maps = pipeline.render_position_multiview(elevs, azims)

    # ── Stage 4: Generate texture views ──────────────────────────────────
    # (deform mode retired — aliased to hybrid at arg-parse time)
    if texture_method == "deform":
        raise RuntimeError("deform mode retired; should have been aliased to hybrid")
    else:
        # ── Stage 4a: Multiview diffusion ─────────────────────────────────
        # The multiview UNet (~3.5 GB) is the biggest CPU RAM consumer. After
        # diffusion we free it so only the bake/inpaint stage remains.
        camera_info = [
            (((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev]
            + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev]
            for azim, elev in zip(azims, elevs)
        ]

        report(58, "Multiview diffusion", "generating texture views")

        # Feed the multiview model its NATIVE conditioning: exactly ONE
        # reference (front). paint-v2-0 was trained single-view, and the
        # vendored multiview_utils hardcodes camera_info_ref=[[0]] — passing
        # four refs labels them ALL as front-camera views, which is
        # out-of-distribution and produced catastrophically corrupted views
        # (e.g. hallucinated backgrounds / holes on the right camera).
        # The multiview model was trained to accept multiple reference views.
        # Pass all available references — the model conditions on them directly.
        _mv_model = pipeline.models["multiview_model"]
        _refs = list(cond_views)
        while len(_refs) < 4:
            _refs.append(cond_views[0])

        # Report per-step progress by wrapping the scheduler's step() method.
        # Map the chosen number of diffusion steps onto the 58->72 progress band so
        # the UI shows live movement instead of a stall.
        _MV_TOTAL_STEPS = TEXTURE_DIFFUSION_STEPS
        _mv_sched = _mv_model.pipeline.scheduler
        _mv_orig_step = _mv_sched.step
        _mv_counter = {"i": 0}

        def _mv_patched_step(*a, **kw):
            _out = _mv_orig_step(*a, **kw)
            _mv_counter["i"] += 1
            _i = _mv_counter["i"]
            _pct = 58 + int(14 * _i / max(1, _MV_TOTAL_STEPS))
            report(min(_pct, 72), "Multiview diffusion", f"step {_i}/{_MV_TOTAL_STEPS}")
            return _out

        _mv_sched.step = _mv_patched_step
        try:
            multiviews = _mv_model(
                _refs, normal_maps + position_maps, camera_info,
                num_inference_steps=TEXTURE_DIFFUSION_STEPS,
            )
        finally:
            _mv_sched.step = _mv_orig_step

        if texture_method == "hybrid":
            # Referenced views = full-res ORIGINAL pixels, warped onto the
            # mesh silhouette with the smooth thin-plate-spline (no optical
            # flow: it smeared low-texture regions). Unreferenced views
            # (top/bottom) keep the diffusion output. Uses the raw snapshots,
            # not the delight views, so baked pixels are real colours.
            report(68, "Hybrid: warping originals onto mesh silhouettes", "thin-plate spline")
            multiviews = _hybrid_warp_multiview(
                _raw_cond_views, multiviews, position_maps, RENDER_RES)
        else:
            # Resize generated views to the render resolution for baking.
            multiviews = [v.resize((RENDER_RES, RENDER_RES)) for v in multiviews]

        # Free the multiview model + intermediates (~4.5 GB CPU RAM released).
        del cond_views, normal_maps, position_maps
        del pipeline.models["multiview_model"]
        gc.collect()
        torch.cuda.empty_cache()
        report(72, "Diffusion done, freed ~4.5 GB", "")

    # Debug: dump generated/back-projected multiviews.  Erase the dark
    # silhouette contour (inherited from the reference or hallucinated by the
    # diffusion model) BEFORE luminance/colour normalisation so its pixels
    # don't skew the subject statistics or get baked as black piping.
    _mv_names = ["front", "left", "back", "right", "top", "bottom"]
    try:
        multiviews = [_remove_silhouette_outline(v) for v in multiviews]
    except Exception as _ol_err:
        print(json.dumps({"type": "log", "message":
            f"[texture] outline cleanup skipped: {_ol_err}"}), flush=True)
    for _i, (_v, _n) in enumerate(zip(multiviews, _mv_names)):
        _v.save(os.path.join(_debug_dir, f"03_multiview_{_i}_{_n}.png"))

    # Debug: quantify per-view diversity.
    try:
        import numpy as _np
        _mv_arrs = [_np.array(m.convert("RGB")).astype(float) for m in multiviews]
        _diffs = []
        for a in range(len(_mv_arrs)):
            for b in range(a + 1, len(_mv_arrs)):
                _diffs.append(_np.abs(_mv_arrs[a] - _mv_arrs[b]).mean())
        _means = [_np.array(m.convert("RGB")).reshape(-1, 3).mean(0) for m in multiviews]
        _avg_diff = sum(_diffs) / len(_diffs) if _diffs else 0
        _log = ["[multiview diversity] avg pairwise RGB diff = %.1f" % _avg_diff]
        for i, n in enumerate(_mv_names):
            _log.append("  %s meanRGB = %s" % (n, _means[i].round(1).tolist()))
        with open(os.path.join(_debug_dir, "06_diversity_report.txt"), "w") as _fh:
            _fh.write("\n".join(_log))
        print("\n".join(_log))
    except Exception as _e:
        print("[diversity report skipped] %s" % _e)

    # ── Preserve view colors/exposure for baking ──────────────────────────
    # The diffusion views already have consistent neutral lighting. Per-view
    # luminance and histogram matching shifted their colors away from the
    # generated views (and from the reference), so preserve the images here.
    # Keep only the subject masks for silhouette edge cleanup below.
    _fg_masks = []
    for _v in multiviews:
        try:
            _m = _subject_silhouette(np.array(_v.convert("RGB")))
            _cov = float(_m.mean())
            _fg_masks.append(_m if 0.02 < _cov < 0.98 else None)
        except Exception:
            _fg_masks.append(None)
    print(json.dumps({"type": "log", "message":
        "[texture] preserving multiview colors and exposure; per-view normalization skipped"}), flush=True)
    # Save views for debug comparison
    for _i, (_v, _n) in enumerate(zip(multiviews, _mv_names)):
        _v.save(os.path.join(_debug_dir, f"03b_bake_input_{_i}_{_n}.png"))

    # ── Background extension: push subject colour outward ─────────────────
    # Texels on the mesh silhouette boundary project onto view pixels right
    # at (and just outside) the subject edge, where anti-aliasing and the
    # flat backdrop live — that is exactly the white/gray piping visible on
    # every silhouette in the render.  Extend the subject's own colour a few
    # pixels into the background (nearest-subject-pixel fill via a distance
    # transform) so boundary texels read real surface colour.
    try:
        import cv2 as _cv2_be
        _ext = max(2, RENDER_RES // 256)
        _new = []
        for _vi, _v in enumerate(multiviews):
            _a = np.array(_v.convert("RGB"))
            _m = _fg_masks[_vi]
            if _m is None:
                try:
                    _m = _subject_silhouette(_a)
                    if not (0.02 < float(_m.mean()) < 0.98):
                        _new.append(_v)
                        continue
                except Exception:
                    _new.append(_v)
                    continue
            # ring = background pixels within _ext of the subject edge
            _ring = (_cv2_be.dilate(_m.astype(np.uint8),
                                    np.ones((3, 3), np.uint8),
                                    iterations=_ext) - _m.astype(np.uint8)).astype(np.uint8) * 255
            if _ring.sum() == 0:
                _new.append(_v)
                continue
            # Telea inpaint fills the ring from its boundaries; pixels next to
            # the subject edge take the subject's colour, which is all the
            # bake's boundary texels ever sample.
            _a = _cv2_be.inpaint(_a, _ring, _ext, _cv2_be.INPAINT_TELEA)
            _new.append(Image.fromarray(_a))
        multiviews = _new
        print(json.dumps({"type": "log", "message":
            f"[texture] subject colour extended {_ext}px into background (silhouette piping fix)"}), flush=True)
    except Exception as _be_err:
        print(json.dumps({"type": "log", "message":
            f"[texture] background extension skipped: {_be_err}"}), flush=True)

    # ── Stage 5: Bake textures + inpaint (runs on GPU) ───────────────────
    report(75, "Baking texture atlas", "merging projected views")
    texture, mask = pipeline.bake_from_multiview(
        multiviews, elevs, azims, view_weights,
        method=pipeline.config.merge_method,
    )
    _tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    _tex_h, _tex_w = texture.shape[:2]
    # Ensure the baked texture atlas matches the requested resolution.
    # The pipeline may internally downscale; if so, resize using PIL.
    if _tex_h != TEXTURE_SIZE or _tex_w != TEXTURE_SIZE:
        _msg = f"Baked texture is {_tex_w}x{_tex_h}, requested {TEXTURE_SIZE}x{TEXTURE_SIZE} — resizing"
        print(json.dumps({"type": "log", "message": _msg}), flush=True)
        _tex_pil = Image.fromarray(_tex_np).resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
        texture = torch.from_numpy(np.array(_tex_pil).astype(np.float32) / 255.0).to(texture.device)

    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    _mh, _mw = mask.shape[:2]
    if _mh != TEXTURE_SIZE or _mw != TEXTURE_SIZE:
        _pil = Image.fromarray(mask_np).resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.NEAREST)
        mask = torch.from_numpy(np.array(_pil).astype(np.float32) / 255.0).to(mask.device).unsqueeze(-1)
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

    # Debug: dump baked texture and mask
    _tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if _tex_np.ndim == 3 and _tex_np.shape[2] in (3, 4):
        Image.fromarray(_tex_np).save(os.path.join(_debug_dir, "04_baked_texture.png"))
    else:
        print(f"[debug] unexpected texture shape {_tex_np.shape}, skipping debug dump")
    Image.fromarray(mask_np).save(os.path.join(_debug_dir, "05_bake_mask.png"))
    print(json.dumps({"type": "log", "message":
        f"[texture] bake hole fraction: {float((mask_np < 128).mean()):.3f} "
        f"(bake_exp={pipeline.config.bake_exp})"}), flush=True)

    # Inpaint is mandatory (fills UV gaps that would otherwise leave the mesh
    # black). Always run it.
    report(88, "Inpainting UV seams", "filling gaps")
    texture = pipeline.texture_inpaint(texture, mask_np)
    _final = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if _final.ndim == 3 and _final.shape[2] in (3, 4):
        Image.fromarray(_final).save(os.path.join(_debug_dir, "06_inpainted_texture.png"))

    # Flatten texture: RETIRED — normalization and delight handle lighting.
    if False:  # flatten_texture removed
        try:
            _final_pil = Image.fromarray(_final)
            _final_pil = _flatten_texture(_final_pil, strength=1.0)
            _final = np.array(_final_pil)
            print(json.dumps({"type": "log", "message":
                "[texture] flatten applied (strength=1.0) — shadows/highlights removed"}), flush=True)
        except Exception as _ft_err:
            print(json.dumps({"type": "log", "message":
                f"[texture] flatten skipped: {_ft_err}"}), flush=True)

    # Do not histogram-match the finished atlas to the flat reference image:
    # UV coordinates have no spatial relationship to photo pixels, and this
    # global pass included background colors and washed out the baked albedo.
    texture = torch.from_numpy(_final.astype(np.float32) / 255.0).to(texture.device)
    print(json.dumps({"type": "log", "message":
        "[texture] final atlas palette preserved; skipped spatially invalid global correction"}), flush=True)

    pipeline.render.set_texture(texture)
    textured_mesh = pipeline.render.save_mesh()

    report(96, "Saving textured mesh", "exporting GLB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    textured_mesh.export(output_path)
    report(100, "Done", "texture complete")
    print(json.dumps({"type": "done", "output_path": output_path}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    _raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    # Accept either an inline JSON string (host passes this) or a path to a
    # .json file (handy for local testing).
    if os.path.isfile(_raw):
        with open(_raw, "r", encoding="utf-8") as _f:
            args = json.load(_f)
    else:
        args = json.loads(_raw)
    try:
        setup_paths(args)
        texture_mesh(args)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_cuda()
