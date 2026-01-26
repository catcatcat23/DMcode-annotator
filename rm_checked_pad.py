#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2

ROOT = "/home/cat/workspace/DMCODE/SNcode/checkedcases"

# 支持的图片后缀
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PADDING = 10  # 四周要裁掉的像素数


def process_image(path: str):
    dirpath, fname = os.path.split(path)
    name, ext = os.path.splitext(fname)

    # 只处理文件名里包含 checked 的
    if "checked" not in name:
        return

    if ext.lower() not in IMG_EXTS:
        return

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] 无法读取: {path}")
        return

    h, w = img.shape[:2]
    if h <= 2 * PADDING or w <= 2 * PADDING:
        print(f"[SKIP] 图片太小，无法裁掉 {PADDING}px 边缘: {path}")
        return

    # 裁掉四周 PADDING 像素
    cropped = img[PADDING : h - PADDING, PADDING : w - PADDING]

    # 文件名中把 checked -> checked_nopad
    new_name = name.replace("checked", "checked_nopad") + ext
    out_path = os.path.join(dirpath, new_name)

    ok = cv2.imwrite(out_path, cropped)
    if ok:
        print(f"[OK] {path} -> {out_path}")
    else:
        print(f"[ERR] 写入失败: {out_path}")


def main():
    total = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            process_image(path)
            total += 1
    print(f"[DONE] 扫描文件数: {total}")


if __name__ == "__main__":
    main()
