import os
import json
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylibdmtx.pylibdmtx import decode


# ========= 工具函数 =========

def sync_dm_code(sync_dm_array, sync_cell_width, border_width):
    """
    根据 cell 矩阵生成 DM 图（跟你 pipeline 里的一样）
    """
    rows, cols = sync_dm_array.shape
    sync_dm_code = np.zeros((rows * sync_cell_width, cols * sync_cell_width), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if sync_dm_array[i, j] == 1:
                sync_dm_code[
                    i * sync_cell_width:(i + 1) * sync_cell_width,
                    j * sync_cell_width:(j + 1) * sync_cell_width
                ] = 255

    sync_dm_code_padded = cv2.copyMakeBorder(
        src=sync_dm_code,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )
    return sync_dm_code_padded


def decode_and_print(img):
    """用 libdmtx 解码当前图像，并打印结果（只是辅助用）"""
    img_u8 = img.astype(np.uint8)
    res = decode(img_u8, max_count=1)
    if len(res) == 0:
        print("  decode: 失败")
    else:
        try:
            content = res[0].data.decode("utf-8", errors="ignore")
        except Exception:
            content = str(res[0].data)
        print("  decode: 成功 ->", content)


def find_dm_image(bad_dir, prefix):
    """找到 prefix 对应的裁剪 dm_image 图（你保存的是 prefix_dm_image_w*h*.jpg）"""
    pattern = os.path.join(bad_dir, f"{prefix}_dm_image_w*h*.jpg")
    files = glob.glob(pattern)
    return files[0] if files else None


# ========= 编辑单个 prefix 的函数 =========

def edit_one_prefix(bad_dir, prefix):
    """
    对一个 prefix 打开交互界面，允许点击修改 cell。
    返回：是否修改过（bool），以及修改后的 sync / encoder 矩阵
    """
    print(f"\n===== 编辑样本：{prefix} =====")

    # 1) 读 meta
    meta_path = os.path.join(bad_dir, f"{prefix}_meta.json")
    if not os.path.exists(meta_path):
        print("  ⚠ 未找到 meta.json，跳过。")
        return False, None, None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    sync_cell_width = int(meta["sync_cell_width"])
    border_width = int(meta["border_width"])

    # 2) 读 sync_array
    sync_array_path = os.path.join(bad_dir, f"{prefix}_sync_dm_array.npy")
    if not os.path.exists(sync_array_path):
        print("  ⚠ 未找到 sync_array.npy，跳过。")
        return False, None, None
    sync_dm_array = np.load(sync_array_path)
    sync_dm_array_orig = sync_dm_array.copy()

    # 3) 读 encoder_array（可能不存在）
    encoder_dm_array = None
    encoder_dm_array_orig = None
    encoder_array_path = os.path.join(bad_dir, f"{prefix}_encoder_dm_array.npy")
    if os.path.exists(encoder_array_path):
        encoder_dm_array = np.load(encoder_array_path)
        encoder_dm_array_orig = encoder_dm_array.copy()
        has_encoder = True
    else:
        has_encoder = False
        print("  ℹ 未找到 encoder_array.npy，本次只编辑 sync。")

    # 4) 读原始 dm_image
    dm_img_path = find_dm_image(bad_dir, prefix)
    if dm_img_path is None:
        print("  ⚠ 找不到 dm_image 图，仍然可以编辑矩阵，但无法显示原图。")
        dm_image = None
    else:
        dm_image = cv2.imread(dm_img_path, cv2.IMREAD_GRAYSCALE)

    # 5) 根据矩阵生成图像
    sync_dm_code_img = sync_dm_code(sync_dm_array, sync_cell_width, border_width)

    encoder_code_img = None
    if has_encoder:
        # encoder 图一般没白边，这里 border_width 设 0 即可
        encoder_code_img = sync_dm_code(encoder_dm_array, sync_cell_width, 0)

    # 6) 搭界面：原图 / sync / encoder
    n_cols = 1 + 1 + (1 if has_encoder else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    if n_cols == 2:
        ax_orig, ax_sync = axes
        ax_enc = None
    else:
        ax_orig, ax_sync, ax_enc = axes

    if dm_image is not None:
        ax_orig.imshow(dm_image, cmap="gray")
        ax_orig.set_title("Original DM (裁剪后)")
    else:
        ax_orig.text(0.5, 0.5, "No DM image", ha="center", va="center")
        ax_orig.set_title("Original DM (缺失)")
    ax_orig.axis("off")

    sync_im = ax_sync.imshow(sync_dm_code_img, cmap="gray")
    ax_sync.set_title("Sync DM (点击翻转)")
    ax_sync.axis("off")

    if has_encoder and ax_enc is not None:
        enc_im = ax_enc.imshow(encoder_code_img, cmap="gray")
        ax_enc.set_title("Encoder DM (点击翻转)")
        ax_enc.axis("off")
    else:
        enc_im = None

    plt.suptitle(prefix, fontsize=10)
    plt.tight_layout()

    modified = {"sync": False, "enc": False}  # 标记是否有改动

    # 7) 注册点击事件
    def on_click(event):
        nonlocal sync_dm_array, encoder_dm_array, sync_dm_code_img, encoder_code_img

        if event.inaxes not in [ax_sync, ax_enc]:
            return

        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        cw = sync_cell_width
        ch = sync_cell_width  # 正方形 cell

        # 点击 sync 图
        if event.inaxes is ax_sync:
            # sync 图有白边，要减掉 border
            x_cell = x - border_width
            y_cell = y - border_width
            if x_cell < 0 or y_cell < 0:
                return

            j = int(x_cell // cw)  # 列
            i = int(y_cell // ch)  # 行

            if not (0 <= i < rows and 0 <= j < cols):
                return

            print(f"[SYNC] 点击 cell (i={i}, j={j}), 原值={sync_dm_array[i, j]}")
            sync_dm_array[i, j] = 1 - sync_dm_array[i, j]
            print(f"       新值={sync_dm_array[i, j]}")

            sync_dm_code_img = sync_dm_code(sync_dm_array, sync_cell_width, border_width)
            sync_im.set_data(sync_dm_code_img)
            fig.canvas.draw_idle()

            modified["sync"] = True
            # 可以按需打开/注释 decode
            # decode_and_print(sync_dm_code_img)

        # 点击 encoder 图
        elif ax_enc is not None and event.inaxes is ax_enc and has_encoder:
            x_cell = x
            y_cell = y

            j = int(x_cell // cw)
            i = int(y_cell // ch)

            if not (0 <= i < rows and 0 <= j < cols):
                return

            print(f"[ENC] 点击 cell (i={i}, j={j}), 原值={encoder_dm_array[i, j]}")
            encoder_dm_array[i, j] = 1 - encoder_dm_array[i, j]
            print(f"      新值={encoder_dm_array[i, j]}")

            encoder_code_img = sync_dm_code(encoder_dm_array, sync_cell_width, 0)
            enc_im.set_data(encoder_code_img)
            fig.canvas.draw_idle()

            modified["enc"] = True
            # decode_and_print(encoder_code_img)

    fig.canvas.mpl_connect("button_press_event", on_click)

    print("  📌 窗口说明：")
    print("    - 左：原图（仅展示）")
    print("    - 中：Sync 图，点击翻转某个 cell")
    if has_encoder:
        print("    - 右：Encoder 图，点击翻转某个 cell")
    print("    - 关闭窗口后，会自动检测是否修改过矩阵，并决定是否覆盖保存。")

    plt.show()  # 阻塞，直到你关掉这个窗口

    # 8) 判断是否真的发生了修改（和原数组对比）
    sync_changed = not np.array_equal(sync_dm_array, sync_dm_array_orig)
    enc_changed = False
    if has_encoder:
        enc_changed = not np.array_equal(encoder_dm_array, encoder_dm_array_orig)

    if not (sync_changed or enc_changed):
        print("  没有检测到修改，跳过保存。")
        return False, None, None

    print("  检测到修改，将覆盖保存对应矩阵/图像。")

    # 重新生成最终图像（防止你手动点错影响原图）
    if sync_changed:
        new_sync_img = sync_dm_code(sync_dm_array, sync_cell_width, border_width)
        # 覆盖保存 sync_array.npy
        np.save(sync_array_path, sync_dm_array)
        # 覆盖保存 sync 图（可选，看你原来怎么命名的）
        sync_img_path = os.path.join(
            bad_dir,
            f"{prefix}_sync_dm_code_w{new_sync_img.shape[1]}h{new_sync_img.shape[0]}.jpg"
        )
        cv2.imwrite(sync_img_path, new_sync_img)
        print(f"    ✅ 已覆盖 sync_array.npy，并写出 {os.path.basename(sync_img_path)}")

    if has_encoder and enc_changed:
        new_enc_img = sync_dm_code(encoder_dm_array, sync_cell_width, 0)
        np.save(encoder_array_path, encoder_dm_array)
        enc_img_path = os.path.join(
            bad_dir,
            f"{prefix}_encoder_dm_edited.jpg"
        )
        cv2.imwrite(enc_img_path, new_enc_img)
        print(f"    ✅ 已覆盖 encoder_array.npy，并写出 {os.path.basename(enc_img_path)}")

    return True, sync_dm_array if sync_changed else None, encoder_dm_array if enc_changed and has_encoder else None


# ========= 批量处理入口 =========

if __name__ == "__main__":
    BAD_DIR = "/home/cat/workspace/DMCODE/SNcode/badcases"  # 修改成你的 badcases 目录

    # 找到所有 *_meta.json，把 prefix 提取出来
    meta_files = sorted(glob.glob(os.path.join(BAD_DIR, "*_meta.json")))
    if not meta_files:
        print("❌ 目录下没有 *_meta.json，确认 pipeline 是否已经保存了这些文件。")
        exit(0)

    prefixes = [os.path.basename(p)[:-len("_meta.json")] for p in meta_files]
    print(f"共找到 {len(prefixes)} 个样本，将逐个弹窗编辑。")

    for idx, prefix in enumerate(prefixes):
        print(f"\n>>>> [{idx+1}/{len(prefixes)}] 处理 {prefix}")
        edit_one_prefix(BAD_DIR, prefix)

    print("\n🎉 全部样本处理完毕。")
