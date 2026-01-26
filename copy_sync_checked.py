import os
import glob
import shutil

# ===== 路径配置 =====
CHECKED_DIR = "/home/cat/workspace/DMCODE/SNcode/checkedcases"
BLACK_ROOT = "/home/cat/workspace/DMCODE/SNcode/flat_src_rect_250113"

def main():
    # 遍历 black 目录下的所有子文件夹
    for folder_name in os.listdir(BLACK_ROOT):
        folder_path = os.path.join(BLACK_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue

        prefix = folder_name  # 文件夹名就是前缀，例如 282192_..._white_BARCODE_253_BARCODE_101
        print(f"\n[INFO] 处理前缀/文件夹: {prefix}")

        # 在 checkedcases 下找形如：任意id_前缀_sync_dm_checked.扩展名
        pattern = os.path.join(CHECKED_DIR, f"*_{prefix}_sync_dm_checked.jpg")
        matches = glob.glob(pattern)

        if not matches:
            print(f"[WARN] 没找到匹配文件: {pattern}")
            continue

        if len(matches) > 1:
            print(f"[WARN] 找到多个匹配，默认取第一条：")
            for m in matches:
                print("   ", m)

        src = matches[0]
        # 保留原扩展名（.png / .jpg 等）
        _, ext = os.path.splitext(src)

        # 目标文件名：去掉前面的数字id，只保留前缀 + _sync_dm_checked
        dst_name = f"{prefix}_sync_dm_checked{ext}"
        dst = os.path.join(folder_path, dst_name)

        os.makedirs(folder_path, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[COPY] {src}  -->  {dst}")

    print("\n[DONE] 全部文件夹处理完成。")

if __name__ == "__main__":
    main()
