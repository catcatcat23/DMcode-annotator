import os
import shutil
import glob

def filter_encoded_files_from_badcases():
    """
    从badcase文件夹中筛选出包含encode文件的文件组，复制到新文件夹
    """
    # ===================== 配置参数（根据你的路径修改）=====================
    # badcase文件夹路径（源路径）
    badcases_dir = "/home/cat/workspace/DMCODE/SNcode/badcases"
    # 筛选后保存的目标文件夹
    target_dir = "/home/cat/workspace/DMCODE/SNcode/badcases_encoded"
    # encode文件的特征后缀（用于识别有encode的文件组）
    encode_file_suffix = "_encoder_dm.png"
    # =====================================================================

    # 创建目标文件夹（不存在则创建）
    os.makedirs(target_dir, exist_ok=True)

    # 1. 遍历badcase文件夹，找出所有包含encode的文件，提取前缀
    encode_files = glob.glob(os.path.join(badcases_dir, f"*{encode_file_suffix}"))
    if not encode_files:
        print(f"⚠️ 在 {badcases_dir} 中未找到任何包含 {encode_file_suffix} 的文件")
        return

    # 2. 提取每个encode文件的前缀（比如 "123_abcdef"）
    prefix_list = []
    for encode_file in encode_files:
        # 提取文件名（去掉路径）
        filename = os.path.basename(encode_file)
        # 去掉encode后缀，得到前缀
        prefix = filename.replace(encode_file_suffix, "")
        prefix_list.append(prefix)

    print(f"✅ 共找到 {len(prefix_list)} 个包含encode的文件组")

    # 3. 根据前缀筛选该组的所有文件，复制到目标文件夹
    copied_count = 0
    for prefix in prefix_list:
        # 匹配该前缀的所有文件（比如 prefix_*.jpg, prefix_*.npy, prefix_*.json 等）
        related_files = glob.glob(os.path.join(badcases_dir, f"{prefix}*"))
        if not related_files:
            print(f"⚠️ 前缀 {prefix} 未匹配到任何文件，跳过")
            continue

        # 复制每个相关文件到目标文件夹
        for file_path in related_files:
            # 目标文件路径
            target_file = os.path.join(target_dir, os.path.basename(file_path))
            # 跳过已存在的文件（避免覆盖）
            if os.path.exists(target_file):
                print(f"ℹ️ 文件已存在，跳过: {os.path.basename(target_file)}")
                continue
            # 复制文件
            shutil.copy2(file_path, target_file)
            copied_count += 1

    # 4. 输出统计结果
    print(f"\n📊 处理完成！")
    print(f"   - 源文件夹: {badcases_dir}")
    print(f"   - 目标文件夹: {target_dir}")
    print(f"   - 筛选出 {len(prefix_list)} 个文件组")
    print(f"   - 共复制 {copied_count} 个文件")

if __name__ == "__main__":
    filter_encoded_files_from_badcases()