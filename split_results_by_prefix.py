import argparse
import os
from pathlib import Path
import shutil


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, required=True,
                    help=r'包含 *_result_pad.png 和 *_result_no_pad.png 的目录，例如 D:\download\u3p_resnet34_voc_mse_ssim_last')
    ap.add_argument("--cat_dir", type=str, required=True,
                    help=r'包含 *_cat.png 的目录，例如 D:\download\u3p_resnet34_voc_mse_ssim_last\u3p_resnet34_voc_mse_ssim_last')
    ap.add_argument("--out_dir", type=str, required=True,
                    help=r'输出目录，例如 D:\download\u3p_resnet34_voc_mse_ssim_last\grouped')
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="copy: 复制到输出目录；move: 移动到输出目录（会改变原目录）")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    cat_dir = Path(args.cat_dir)
    out_dir = Path(args.out_dir)

    pad_out = out_dir / "pad"
    nopad_out = out_dir / "no_pad"
    safe_mkdir(pad_out)
    safe_mkdir(nopad_out)

    cat_files = sorted(cat_dir.glob("*_cat.png"))
    if not cat_files:
        print(f"[ERROR] cat_dir 下没有找到 *_cat.png: {cat_dir}")
        return

    found_pad = 0
    found_nopad = 0
    missing_pad = 0
    missing_nopad = 0

    pad_list = []
    nopad_list = []

    def do_copy_or_move(src: Path, dst_dir: Path):
        dst = dst_dir / src.name
        if args.mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return dst

    for cat_path in cat_files:
        prefix = cat_path.name[:-len("_cat.png")]  # 去掉后缀得到前缀

        pad_name = f"{prefix}_result_pad.png"
        nopad_name = f"{prefix}_result_no_pad.png"

        pad_src = result_dir / pad_name
        nopad_src = result_dir / nopad_name

        if pad_src.exists():
            dst = do_copy_or_move(pad_src, pad_out)
            pad_list.append(str(dst))
            found_pad += 1
        else:
            missing_pad += 1

        if nopad_src.exists():
            dst = do_copy_or_move(nopad_src, nopad_out)
            nopad_list.append(str(dst))
            found_nopad += 1
        else:
            missing_nopad += 1

    # 写清单
    (out_dir / "pad_list.txt").write_text("\n".join(pad_list), encoding="utf-8")
    (out_dir / "no_pad_list.txt").write_text("\n".join(nopad_list), encoding="utf-8")

    print("[DONE]")
    print(f"cat files: {len(cat_files)}")
    print(f"found pad: {found_pad}, missing pad: {missing_pad}")
    print(f"found no_pad: {found_nopad}, missing no_pad: {missing_nopad}")
    print(f"output: {out_dir}")
    print(f"  - {pad_out}")
    print(f"  - {nopad_out}")
    print(f"  - pad_list.txt / no_pad_list.txt")


if __name__ == "__main__":
    main()
