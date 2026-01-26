#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from collections import defaultdict

TASK_FOLDER = "/home/cat/workspace/DMCODE/SNcode"
CHECKED_DIR = os.path.join(TASK_FOLDER, "checkedcases")
SRC_DIR = os.path.join(TASK_FOLDER, "badcases")

OUT_ORI_SYNC = os.path.join(CHECKED_DIR, "ori_sync")
OUT_PAD10_SYNC = os.path.join(CHECKED_DIR, "ori_pad10_sync")

# ✅ 只要这个后缀的 sync
ONLY_SYNC_SUFFIX = "_sync_dm_checked_nopad"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def build_index_by_stem(root_dir: str):
    """stem(无扩展名) -> [fullpath...]"""
    idx = defaultdict(list)
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(IMG_EXTS):
                stem = os.path.splitext(fn)[0]
                idx[stem].append(os.path.join(dp, fn))
    for k in idx:
        idx[k].sort()
    return idx


def extract_key_from_checked(fn: str):
    """
    只处理: <prefix>_sync_dm_checked_nopad.*
    prefix 里如果是 '0_xxx' 这种，去掉最前面的 '0_'
    返回 key (你要匹配 ori/ori_pad10 的那段)
    """
    stem, _ = os.path.splitext(fn)
    if ONLY_SYNC_SUFFIX not in stem:
        return None

    pos = stem.find("_sync_dm")
    if pos < 0:
        return None

    prefix = stem[:pos]  # e.g. 0_254388_..._101  或 223786_..._101



    return prefix


def extract_leading_int(fn: str):
    """
    从文件名里取开头的数字 id（如 0_xxx... -> 0），没有则返回 None
    """
    stem = os.path.splitext(fn)[0]
    m = re.match(r"^(\d+)_", stem)
    return int(m.group(1)) if m else None


def make_unique_path(dst_path: str):
    """若目标已存在，自动加 __1/__2..."""
    if not os.path.exists(dst_path):
        return dst_path
    base, ext = os.path.splitext(dst_path)
    k = 1
    while True:
        cand = f"{base}__{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def copy2_flat(src: str, dst_dir: str, dst_name: str):
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, dst_name)
    dst_path = make_unique_path(dst_path)
    shutil.copy2(src, dst_path)
    return dst_path


def main():
    os.makedirs(OUT_ORI_SYNC, exist_ok=True)
    os.makedirs(OUT_PAD10_SYNC, exist_ok=True)

    print("[1/3] Indexing source:", SRC_DIR)
    src_idx = build_index_by_stem(SRC_DIR)
    print("  indexed:", sum(len(v) for v in src_idx.values()), "files")

    print("[2/3] Collecting ONLY nopad sync from:", CHECKED_DIR)

    # ✅ 先收集：key -> [候选 sync 路径...]
    key_to_sync_candidates = defaultdict(list)

    for dp, dirs, fns in os.walk(CHECKED_DIR):
        # ✅ 跳过输出目录，避免重复扫导出的文件
        dirs[:] = [d for d in dirs if not d.startswith("export_")]

        for fn in fns:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(fn)[0]
            if not stem.endswith(ONLY_SYNC_SUFFIX):
                continue

            key = extract_key_from_checked(fn)
            if not key:
                continue

            key_to_sync_candidates[key].append(os.path.join(dp, fn))

    keys = sorted(key_to_sync_candidates.keys())
    print("  unique keys (with nopad sync):", len(keys))

    # ✅ key 去重：每个 key 只选一个 sync
    key_to_sync = {}
    dup_keys = 0
    for key in keys:
        cands = key_to_sync_candidates[key]
        if len(cands) > 1:
            dup_keys += 1

        # 规则：优先选“文件名开头数字id最小”的；没有数字则按路径排序取第一个
        def sort_key(p):
            fn = os.path.basename(p)
            lead = extract_leading_int(fn)
            # lead None -> 放后面
            return (lead is None, lead if lead is not None else 10**18, fn)

        cands_sorted = sorted(cands, key=sort_key)
        key_to_sync[key] = cands_sorted[0]

    print("  keys with multiple nopad sync (deduped):", dup_keys)

    print("[3/3] Exporting flattened ori+sync ...")

    total = len(keys)
    ok = 0
    miss = 0
    miss_ori = 0
    miss_pad = 0

    for key in keys:
        sync_path = key_to_sync[key]
        sync_ext = os.path.splitext(sync_path)[1]

        # 输出 sync 文件名统一用：key + _sync_dm_checked_nopad + ext
        sync_out_name = f"{key}{ONLY_SYNC_SUFFIX}{sync_ext}"

        ori_stem = f"{key}_ori"
        pad_stem = f"{key}_ori_pad10"

        ori_list = src_idx.get(ori_stem, [])
        pad_list = src_idx.get(pad_stem, [])

        found_any = False

        # A: ori + sync
        if ori_list:
            ori_src = ori_list[0]
            ori_ext = os.path.splitext(ori_src)[1]
            copy2_flat(ori_src, OUT_ORI_SYNC, f"{key}_ori{ori_ext}")
            copy2_flat(sync_path, OUT_ORI_SYNC, sync_out_name)
            found_any = True
        else:
            miss_ori += 1

        # B: ori_pad10 + sync
        if pad_list:
            pad_src = pad_list[0]
            pad_ext = os.path.splitext(pad_src)[1]
            copy2_flat(pad_src, OUT_PAD10_SYNC, f"{key}_ori_pad10{pad_ext}")
            copy2_flat(sync_path, OUT_PAD10_SYNC, sync_out_name)
            found_any = True
        else:
            miss_pad += 1

        if found_any:
            ok += 1
        else:
            miss += 1
            print(f"[MISS] key={key} | sync={os.path.basename(sync_path)} (no ori and no pad10)")

    print("\n=== DONE ===")
    print("unique keys:", total)
    print("matched (has ori or pad10):", ok)
    print("miss (no ori and no pad10):", miss)
    print("miss_ori:", miss_ori, "miss_pad10:", miss_pad)
    print("OUT_ORI_SYNC:", OUT_ORI_SYNC)
    print("OUT_PAD10_SYNC:", OUT_PAD10_SYNC)


if __name__ == "__main__":
    main()
