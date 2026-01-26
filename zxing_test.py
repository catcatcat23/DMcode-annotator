from pathlib import Path
import csv
import shutil
import cv2
import zxingcpp


def _read_one(img_bgr):
    """兼容不同版本的 zxingcpp API：read_barcode / read_barcodes"""
    if img_bgr is None:
        return None

    # 工业码通常灰度更稳
    if len(img_bgr.shape) == 3:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = img_bgr

    if hasattr(zxingcpp, "read_barcode"):
        return zxingcpp.read_barcode(img)

    if hasattr(zxingcpp, "read_barcodes"):
        rs = zxingcpp.read_barcodes(img)
        return rs[0] if rs else None

    raise RuntimeError("zxingcpp has no read_barcode/read_barcodes")


def _copy_to_fail(p: Path, in_root: Path, fail_root: Path):
    """按相对路径复制，避免重名覆盖"""
    rel = p.relative_to(in_root)
    dst = fail_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    return dst


def main(img_dir: str, pattern="*result.png", recursive=True, fail_dir="zxingcpp_failed"):
    in_root = Path(img_dir)
    fail_root = Path(fail_dir)
    fail_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(in_root.rglob(pattern) if recursive else in_root.glob(pattern))
    print(f"[INFO] dir={in_root}")
    print(f"[INFO] found {len(paths)} files with pattern={pattern}")
    print(f"[INFO] fail_dir={fail_root.resolve()}")

    out_csv = in_root / "zxingcpp_decode_results.csv"
    fail_csv = fail_root / "failed_cases.csv"

    ok_cnt = 0
    miss_cnt = 0
    err_cnt = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out, \
         open(fail_csv, "w", newline="", encoding="utf-8") as f_fail:
        w_out = csv.writer(f_out)
        w_fail = csv.writer(f_fail)

        w_out.writerow(["path", "ok", "format", "text", "error"])
        w_fail.writerow(["path", "reason", "copied_to"])

        for p in paths:
            try:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    miss_cnt += 1
                    dst = _copy_to_fail(p, in_root, fail_root)
                    w_out.writerow([str(p), 0, "", "", "imread failed"])
                    w_fail.writerow([str(p), "imread failed", str(dst)])
                    print(f"[MISS] {p}  (imread failed)")
                    continue

                r = _read_one(img)
                if r is None:
                    miss_cnt += 1
                    dst = _copy_to_fail(p, in_root, fail_root)
                    w_out.writerow([str(p), 0, "", "", "no result"])
                    w_fail.writerow([str(p), "no result", str(dst)])
                    print(f"[MISS] {p}")
                    continue

                text = getattr(r, "text", "") or getattr(r, "raw", "") or ""
                fmt = str(getattr(r, "format", ""))

                if not text:
                    miss_cnt += 1
                    dst = _copy_to_fail(p, in_root, fail_root)
                    w_out.writerow([str(p), 0, fmt, "", "empty text"])
                    w_fail.writerow([str(p), "empty text", str(dst)])
                    print(f"[MISS] {p}  (empty text)")
                    continue

                ok_cnt += 1
                w_out.writerow([str(p), 1, fmt, text, ""])
                print(f"[OK]   {p.name}\t{fmt}\t{text}")

            except Exception as e:
                err_cnt += 1
                # 异常也当失败，复制出来
                dst = _copy_to_fail(p, in_root, fail_root)
                reason = f"{type(e).__name__}: {e}"
                w_out.writerow([str(p), 0, "", "", reason])
                w_fail.writerow([str(p), reason, str(dst)])
                print(f"[ERR]  {p}  {reason}")

    print(f"[DONE] ok={ok_cnt}, miss={miss_cnt}, err={err_cnt}")
    print(f"[DONE] results_csv={out_csv}")
    print(f"[DONE] failed_csv={fail_csv}")
    print(f"[DONE] failed_imgs_root={fail_root.resolve()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir")
    ap.add_argument("--pattern", default="*result.png")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--fail-dir", default="zxingcpp_failed")
    args = ap.parse_args()

    main(
        args.img_dir,
        pattern=args.pattern,
        recursive=(not args.no_recursive),
        fail_dir=args.fail_dir,
    )
