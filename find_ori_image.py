#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import numpy as np

# =========================================================
# 默认配置：不需要手动传参
# =========================================================

# ✅ result 目录：递归扫描这里的所有 *_result_no_pad.png
RESULT_DIR = "/home/cat/workspace/DMCODE/SNcode/泰国"

# ✅ white 原图 + txt 所在根目录（递归找）
# 例：
#   {prefix}_insp_white.png
#   {prefix}_insp_white.txt
ORI_WHITE_ROOT = "/home/cat/workspace/DMCODE/backup/task_69258/task_69258"

# ✅ pad 外扩比例（10%）
EXPAND_RATIO_PAD = 0.10

# ✅ pad 扩大后是否把四点裁剪到图像范围内（避免越界“补边”）
CLIP_TO_IMAGE = True

# 输出文件夹（会建在 RESULT_DIR 下面）
OUT_NO_PAD_DIRNAME = "no_pad"
OUT_PAD_DIRNAME = "pad"

# 结果 mask 可能和 ori 尺寸不同，需要坐标缩放到 mask 坐标系
SCALE_PTS_FOR_MASK_IF_SHAPE_DIFF = True


# =========================================================
# 工具函数
# =========================================================

def extract_prefix_from_result_filename(result_path: str) -> str:
    """
    例：
      0a3b..._insp_white_rows22_cols22_result_no_pad.png
    prefix = 0a3b...
    """
    fn = os.path.basename(result_path)
    if "_insp_" in fn:
        return fn.split("_insp_", 1)[0]
    return fn.split("_", 1)[0]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def order_points(pts: np.ndarray) -> np.ndarray:
    """将四点排序为：tl, tr, br, bl"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def quad_size(pts_ordered: np.ndarray):
    """根据四边形估计输出 w,h"""
    tl, tr, br, bl = pts_ordered
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    w = max(1, int(round(max(w1, w2))))
    h = max(1, int(round(max(h1, h2))))
    return w, h


def warp_quad(image: np.ndarray, pts: np.ndarray):
    """
    对四边形区域做透视矫正裁剪，输出矩形 patch。
    注意：这里不是“加边框”，只是把四边形映射成矩形。
    """
    pts_ord = order_points(pts)
    w, h = quad_size(pts_ord)

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_ord, dst)
    patch = cv2.warpPerspective(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # 只有越界才会用到；我们默认会 clip，正常不会出现“补边”
    )
    return patch


def expand_quad(pts: np.ndarray, ratio: float) -> np.ndarray:
    """
    以中心点为基准扩大四点框：
      pts' = c + (pts-c)*(1+ratio)
    这就是“多扣一点”，不是 padding。
    """
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0, keepdims=True)
    scale = 1.0 + float(ratio)
    return c + (pts - c) * scale


def clip_pts_to_image(pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """把点坐标 clip 到图像范围内，避免扩大后越界导致的填充边"""
    pts = np.asarray(pts, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
    return pts


def scale_pts_to_other_image(pts: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    """将坐标从 src 图坐标系映射到 dst 图坐标系"""
    sw, sh = src_wh
    dw, dh = dst_wh
    sx = dw / float(sw)
    sy = dh / float(sh)
    pts = np.asarray(pts, dtype=np.float32).copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts


def find_white_png_and_txt(prefix: str):
    """递归找 {prefix}_insp_white.png / {prefix}_insp_white.txt"""
    png_name = f"{prefix}_insp_white.png"
    txt_name = f"{prefix}_insp_white.txt"
    png_candidates = glob.glob(os.path.join(ORI_WHITE_ROOT, "**", png_name), recursive=True)
    txt_candidates = glob.glob(os.path.join(ORI_WHITE_ROOT, "**", txt_name), recursive=True)
    png_path = sorted(png_candidates)[0] if png_candidates else None
    txt_path = sorted(txt_candidates)[0] if txt_candidates else None
    return png_path, txt_path


def read_quad_from_txt(txt_path: str) -> np.ndarray:
    """
    txt 格式：
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult
    0 0 929 0 929 953 0 953 COMPONENT 0
    取第一条有效数据行的前 8 个数字
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        raise ValueError(f"txt为空：{txt_path}")

    data_line = None
    for ln in lines:
        toks = ln.split()
        if len(toks) >= 8 and toks[0].lstrip("-").isdigit():
            data_line = ln
            break

    if data_line is None:
        raise ValueError(f"未找到有效数据行：{txt_path}")

    toks = data_line.split()
    nums = list(map(float, toks[:8]))
    pts = np.array(nums, dtype=np.float32).reshape(4, 2)
    return pts


# =========================================================
# 主处理逻辑
# =========================================================

def process_one_result(result_path: str, out_no_pad: str, out_pad: str):
    result_path = os.path.abspath(result_path)
    prefix = extract_prefix_from_result_filename(result_path)

    print(f"\n[INFO] prefix = {prefix}")
    print(f"[INFO] result = {result_path}")

    ori_png, ori_txt = find_white_png_and_txt(prefix)
    if ori_png is None or ori_txt is None:
        print(f"[SKIP] 找不到 white png 或 txt：png={ori_png}, txt={ori_txt}")
        return

    # 读 white 原图（你说只需要这个作为 ori）
    ori = cv2.imread(ori_png, cv2.IMREAD_COLOR)
    if ori is None:
        print(f"[SKIP] ori 读取失败：{ori_png}")
        return

    # 读 result mask（你之前要求两两配对，因此仍裁 mask；如果你不需要 mask 我也能删掉）
    mask = cv2.imread(result_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[SKIP] result 读取失败：{result_path}")
        return

    # 读 txt 四点（基于 ori 坐标系）
    try:
        pts_ori = read_quad_from_txt(ori_txt)
    except Exception as e:
        print(f"[SKIP] 读取 txt 失败：{e}")
        return

    oh, ow = ori.shape[:2]
    mh, mw = mask.shape[:2]

    # mask 坐标系的四点（若尺寸不同则缩放）
    pts_mask = pts_ori
    if SCALE_PTS_FOR_MASK_IF_SHAPE_DIFF and (ow, oh) != (mw, mh):
        pts_mask = scale_pts_to_other_image(pts_ori, src_wh=(ow, oh), dst_wh=(mw, mh))
        print(f"[WARN] mask尺寸与ori不一致：ori=({ow},{oh}) mask=({mw},{mh})，已将pts缩放到mask坐标系")

    # ------------------------
    # no_pad：严格按 txt 裁剪
    # ------------------------
    pts_ori_no = pts_ori
    pts_mask_no = pts_mask

    if CLIP_TO_IMAGE:
        pts_ori_no = clip_pts_to_image(pts_ori_no, ow, oh)
        pts_mask_no = clip_pts_to_image(pts_mask_no, mw, mh)

    ori_no_pad = warp_quad(ori, pts_ori_no)
    mask_no_pad = warp_quad(mask, pts_mask_no)

    ori_no_pad_path = os.path.join(out_no_pad, f"{prefix}_ori_no_pad.png")
    mask_no_pad_path = os.path.join(out_no_pad, f"{prefix}_result_no_pad_mask.png")
    cv2.imwrite(ori_no_pad_path, ori_no_pad)
    cv2.imwrite(mask_no_pad_path, mask_no_pad)

    # ------------------------
    # pad：把 txt 框扩大 10% 再裁剪（不是补白边）
    # ------------------------
    pts_ori_pad = expand_quad(pts_ori, EXPAND_RATIO_PAD)
    pts_mask_pad = expand_quad(pts_mask, EXPAND_RATIO_PAD)

    if CLIP_TO_IMAGE:
        pts_ori_pad = clip_pts_to_image(pts_ori_pad, ow, oh)
        pts_mask_pad = clip_pts_to_image(pts_mask_pad, mw, mh)

    ori_pad = warp_quad(ori, pts_ori_pad)
    mask_pad = warp_quad(mask, pts_mask_pad)

    ori_pad_path = os.path.join(out_pad, f"{prefix}_ori_pad.png")
    mask_pad_path = os.path.join(out_pad, f"{prefix}_result_pad_mask.png")
    cv2.imwrite(ori_pad_path, ori_pad)
    cv2.imwrite(mask_pad_path, mask_pad)

    print("[OK] 输出：")
    print(f"  no_pad: {ori_no_pad_path}")
    print(f"  no_pad: {mask_no_pad_path}")
    print(f"  pad   : {ori_pad_path}")
    print(f"  pad   : {mask_pad_path}")


def main():
    root = os.path.abspath(RESULT_DIR)
    print(f"[START] 扫描 result 目录：{root}")
    print(f"[INFO]  white ori+txt 根目录：{ORI_WHITE_ROOT}")
    print(f"[INFO]  pad 外扩比例：{EXPAND_RATIO_PAD * 100:.1f}%")
    print(f"[INFO]  CLIP_TO_IMAGE={CLIP_TO_IMAGE}")

    out_no_pad = os.path.join(root, OUT_NO_PAD_DIRNAME)
    out_pad = os.path.join(root, OUT_PAD_DIRNAME)
    ensure_dir(out_no_pad)
    ensure_dir(out_pad)

    # 递归找所有 png，但跳过输出目录 no_pad/pad
    all_png = []
    for dp, _, fns in os.walk(root):
        base = os.path.basename(dp).lower()
        if base in (OUT_NO_PAD_DIRNAME.lower(), OUT_PAD_DIRNAME.lower()):
            continue
        for fn in fns:
            if fn.lower().endswith(".png"):
                all_png.append(os.path.join(dp, fn))

    # 只处理真正的结果图，避免把你生成的输出也处理进去
    results = [p for p in sorted(all_png) if p.lower().endswith("_result_no_pad.png")]

    print(f"[INFO] 找到 {len(results)} 个 *_result_no_pad.png")
    if not results:
        print("[WARN] 没找到 *_result_no_pad.png。检查 result 图命名是否包含 _result_no_pad.png")
        return

    for rp in results:
        process_one_result(rp, out_no_pad=out_no_pad, out_pad=out_pad)

    print("[DONE] 全部处理完成")


if __name__ == "__main__":
    main()
