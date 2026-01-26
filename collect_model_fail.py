import os
import glob
import shutil

# ====== 配置区 ======
BASE_DIR = "/home/cat/workspace/DMCODE/SNcode/DM_code20251203_SHMS_provide_insp_crop_data"
OUT_DIR = "/home/cat/workspace/DMCODE/SNcode/DM_code20251203_SHMS_provide_insp_crop_data_model_fail"

# 这些就是你发的前缀（不带扩展名）
BAD_PREFIXES = [
'404198_a8b6deb3-40fb-4747-a8b0-3cb8201d03a0_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404201_5db26c7d-a2f4-40cd-ab19-74bd52920070_white_EXTERNAL_BARCODE_201_EXTERNAL_BARCODE_101',
'404202_258255fe-218c-410f-9777-e2ac35705e67_white_EXTERNAL_BARCODE_201_EXTERNAL_BARCODE_101',
'404202_258255fe-218c-410f-9777-e2ac35705e67_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
'404204_fcb13f20-e176-4810-b6ab-1d8351ba077f_white_EXTERNAL_BARCODE_201_EXTERNAL_BARCODE_101',
'404205_18c1babf-6ad0-4061-aa16-640c97e27919_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
'404205_18c1babf-6ad0-4061-aa16-640c97e27919_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404206_421f2108-d0da-4482-8912-4c2f1aa7ce83_white_EXTERNAL_BARCODE_200_EXTERNAL_BARCODE_101',
'404206_421f2108-d0da-4482-8912-4c2f1aa7ce83_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404210_b05005ff-d827-4196-82b5-cd4bfda0f883_white_EXTERNAL_BARCODE_200_EXTERNAL_BARCODE_101',
'404210_b05005ff-d827-4196-82b5-cd4bfda0f883_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404213_87272b6f-3708-4e2c-99e0-2548944d203b_white_EXTERNAL_BARCODE_200_EXTERNAL_BARCODE_101',
'404213_87272b6f-3708-4e2c-99e0-2548944d203b_white_EXTERNAL_BARCODE_201_EXTERNAL_BARCODE_101',
'404213_87272b6f-3708-4e2c-99e0-2548944d203b_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
'404213_87272b6f-3708-4e2c-99e0-2548944d203b_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404215_a5e31c1d-a10b-48ec-982d-4aa21f51608d_white_EXTERNAL_BARCODE_201_EXTERNAL_BARCODE_101',
'404215_a5e31c1d-a10b-48ec-982d-4aa21f51608d_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
'404217_a5e08eb9-47f3-423b-ba3e-6e1b05b3b7d0_white_EXTERNAL_BARCODE_200_EXTERNAL_BARCODE_101',
'404217_a5e08eb9-47f3-423b-ba3e-6e1b05b3b7d0_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
'404217_a5e08eb9-47f3-423b-ba3e-6e1b05b3b7d0_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404218_b7189756-6fec-440a-94fc-dddeb329d84b_white_EXTERNAL_BARCODE_200_EXTERNAL_BARCODE_101',
'404218_b7189756-6fec-440a-94fc-dddeb329d84b_white_EXTERNAL_BARCODE_201_EXTERNAL_BARCODE_101',
'404218_b7189756-6fec-440a-94fc-dddeb329d84b_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
'404218_b7189756-6fec-440a-94fc-dddeb329d84b_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404219_5e8fe199-82be-4693-b34b-5304fdc482e0_white_EXTERNAL_BARCODE_203_EXTERNAL_BARCODE_101',
'404220_b00dba90-b329-48e7-ab07-bf08fd7e21fe_white_EXTERNAL_BARCODE_202_EXTERNAL_BARCODE_101',
]

# ====== 代码区 ======
os.makedirs(OUT_DIR, exist_ok=True)

def copy_matches(prefix: str):
    found_any = False
    for ext in (".png", ".json"):
        # 假设文件名是  prefix*.png / prefix*.json，递归全目录找
        pattern = os.path.join(BASE_DIR, "**", prefix + "*" + ext)
        for src in glob.glob(pattern, recursive=True):
            found_any = True
            dst = os.path.join(OUT_DIR, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"[COPY] {src}  ->  {dst}")
    if not found_any:
        print(f"[WARN] 没找到任何文件：前缀 = {prefix}")

def main():
    print(f"BASE_DIR = {BASE_DIR}")
    print(f"OUT_DIR  = {OUT_DIR}")
    print(f"前缀数量: {len(BAD_PREFIXES)}")
    print("=" * 60)

    for prefix in BAD_PREFIXES:
        copy_matches(prefix)

    print("=" * 60)
    print("完成。可以到 OUT_DIR 目录里检查结果。")

if __name__ == "__main__":
    main()
