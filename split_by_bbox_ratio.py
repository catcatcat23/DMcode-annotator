#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

from tqdm.auto import tqdm  # <- 进度条

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path, dry_run: bool):
    safe_mkdir(dst.parent)
    if dry_run:
        return
    shutil.copy2(str(src), str(dst))

def pick_main_annotation(ann):
    if isinstance(ann, dict):
        return ann
    if isinstance(ann, list) and len(ann) > 0:
        def score(a):
            if isinstance(a, dict):
                if "area" in a and isinstance(a["area"], (int, float)):
                    return float(a["area"])
                bb = a.get("bbox", None)
                if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                    w, h = float(bb[2]), float(bb[3])
                    return abs(w * h)
            return -1.0
        return max(ann, key=score)
    return None

def read_bbox_wh(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise KeyError("missing key: annotations")

    a = pick_main_annotation(data["annotations"])
    if not isinstance(a, dict):
        raise ValueError("annotations format not supported")

    bb = a.get("bbox", None)
    if not (isinstance(bb, (list, tuple)) and len(bb) >= 4):
        raise ValueError("missing/invalid bbox")

    w = float(bb[2])
    h = float(bb[3])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid bbox w/h: {w}, {h}")
    return w, h

def is_rect_by_bbox(json_path: Path, thr: float) -> bool:
    w, h = read_bbox_wh(json_path)
    ratio = max(w / h, h / w)
    return ratio > thr

def find_image_by_stem(stem_path: Path):
    for ext in IMG_EXTS:
        p = stem_path.with_suffix(ext)
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Split images+json by bbox aspect ratio threshold (copy only).")
    ap.add_argument("in_dir", type=str, help="输入目录（包含图片和同名json）")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="输出根目录（默认: <in_dir>/_split_bbox_ratio）")
    ap.add_argument("--thr", type=float, default=1.05,
                    help="长宽比阈值：ratio>thr 归为长方形（默认1.05）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument("--dry_run", action="store_true", help="只打印不执行")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (in_dir / "_split_bbox_ratio")
    sq_dir = out_dir / "square"
    rect_dir = out_dir / "rect"
    safe_mkdir(sq_dir)
    safe_mkdir(rect_dir)

    it = in_dir.rglob("*.json") if args.recursive else in_dir.glob("*.json")
    json_files = [p for p in it if p.is_file()]  # <- 先收集成 list，tqdm 才能显示总数

    total = len(json_files)
    sq_cnt = rect_cnt = miss_img = bad_json = 0

    pbar = tqdm(json_files, desc="Split by bbox", unit="json", dynamic_ncols=True)
    for j in pbar:
        try:
            is_rect = is_rect_by_bbox(j, thr=args.thr)
        except Exception as e:
            bad_json += 1
            # 不刷屏的话可以注释下一行
            # print(f"[WARN] bad json: {j} ({e})")
            pbar.set_postfix(sq=sq_cnt, rect=rect_cnt, miss=miss_img, bad=bad_json)
            continue

        target_dir = rect_dir if is_rect else sq_dir
        if is_rect:
            rect_cnt += 1
        else:
            sq_cnt += 1

        dst_json = target_dir / j.name
        if args.dry_run:
            pass
        else:
            copy_file(j, dst_json, dry_run=False)

        img = find_image_by_stem(j.with_suffix(""))
        if img is None:
            miss_img += 1
            pbar.set_postfix(sq=sq_cnt, rect=rect_cnt, miss=miss_img, bad=bad_json)
            continue

        dst_img = target_dir / img.name
        if args.dry_run:
            pass
        else:
            copy_file(img, dst_img, dry_run=False)

        # 实时显示统计
        pbar.set_postfix(sq=sq_cnt, rect=rect_cnt, miss=miss_img, bad=bad_json)

    print("\n=== DONE ===")
    print(f"in_dir     : {in_dir}")
    print(f"out_dir    : {out_dir}")
    print(f"thr        : {args.thr} (ratio>thr => rect)")
    print("mode       : COPY (will NOT modify source folder)")
    print(f"total json : {total}")
    print(f"square dir : {sq_cnt}")
    print(f"rect dir   : {rect_cnt}")
    print(f"miss image : {miss_img}")
    print(f"bad json   : {bad_json}")

if __name__ == "__main__":
    main()
