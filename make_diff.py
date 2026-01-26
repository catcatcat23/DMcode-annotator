#!/usr/bin/env python3
import os
from PIL import Image, ImageChops

# 根目录
ROOT = "/home/cat/workspace/DMCODE/DA-2444_2_no_pad_results"

ENCODED_NAME = "encoded_no_pad.png"
SYNC_NAME = "sync_dm_no_pad.png"
DIFF_NAME = "diff.png"

def make_diff(encoded_path, sync_path, save_path, threshold=10):
    """
    encoded_path: encoded_no_pad.png
    sync_path   : sync_dm_no_pad.png
    save_path   : diff.png
    threshold   : 差异阈值，避免小噪点

    return:
        has_diff: bool，是否存在差异（超过阈值）
    """
    img1 = Image.open(encoded_path).convert("RGB")
    img2 = Image.open(sync_path).convert("RGB")

    if img1.size != img2.size:
        # 如果大小不一样，简单缩放第二张到第一张大小
        img2 = img2.resize(img1.size)

    # 计算像素差
    diff = ImageChops.difference(img1, img2)  # RGB 差值

    # 转灰度做 mask
    mask = diff.convert("L")

    # 阈值化：小于 threshold 的差异忽略掉
    def _th(v):
        return 255 if v > threshold else 0
    mask = mask.point(_th)

    # 检查是否真的有差异
    bbox = mask.getbbox()
    has_diff = bbox is not None

    if has_diff:
        # 用红色高亮差异区域
        red_layer = Image.new("RGB", img1.size, (255, 0, 0))
        out = img1.copy()
        out.paste(red_layer, mask=mask)

        out.save(save_path)
        print(f"[OK] diff saved: {save_path}")
    else:
        print(f"[SAME] no significant diff between:\n       {encoded_path}\n       {sync_path}")

    return has_diff

def main():
    diff_dirs = []  # 记录有差异的“子目录名”

    for dirpath, dirnames, filenames in os.walk(ROOT):
        if ENCODED_NAME in filenames and SYNC_NAME in filenames:
            encoded_path = os.path.join(dirpath, ENCODED_NAME)
            sync_path = os.path.join(dirpath, SYNC_NAME)
            diff_path = os.path.join(dirpath, DIFF_NAME)

            try:
                has_diff = make_diff(encoded_path, sync_path, diff_path)
                if has_diff:
                    # 只记录子目录名，不要全路径
                    basename = os.path.basename(dirpath)
                    diff_dirs.append(basename)
            except Exception as e:
                print(f"[ERR] {dirpath}: {e}")

    # 去重 + 排序（可选）
    diff_dirs = sorted(set(diff_dirs))

    print("\n========== Subdirectories with differences ==========")
    if not diff_dirs:
        print("No directories with differences found.")
    else:
        for name in diff_dirs:
            print(name)
        print(f"\nTotal: {len(diff_dirs)} subdirectories with differences.")

if __name__ == "__main__":
    main()
