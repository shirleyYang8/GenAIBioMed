#!/usr/bin/env python3
"""
decompress_zips.py
- Decompresses all .zip files in a source directory (optionally recursively).
- Uses only Python standard library.
- Safety: prevents zip-slip (no writing outside the output directory).

Usage:
  python decompress_zip.py --src /path/to/zips --out /path/to/extracted

Notes:
- Each ZIP is extracted to: <out>/<zip_stem>/
- Non-ZIP files are ignored. Corrupt ZIPs are reported and skipped.
"""

import argparse
import sys
from pathlib import Path
import zipfile

def find_zip_files(src: Path, recursive: bool):
    pattern = "**/*.zip" if recursive else "*.zip"
    return sorted(p for p in src.glob(pattern) if p.is_file())

def safe_extract(zf: zipfile.ZipFile, target_dir: Path):
    """
    Extracts ZIP contents into target_dir while preventing zip-slip.
    """
    for member in zf.infolist():
        # Resolve the final path to ensure it's within target_dir
        dest_path = (target_dir / member.filename).resolve()
        if not str(dest_path).startswith(str(target_dir.resolve()) + str(Path().anchor if Path().anchor else "")) \
           and target_dir.drive == dest_path.drive:
            # Fallback check (handles drive letters on Windows)
            pass
        if target_dir.resolve() not in dest_path.parents and dest_path != target_dir.resolve():
            # If the resolved destination is not inside target_dir, skip it
            raise RuntimeError(f"Blocked potential zip-slip: {member.filename}")

        if member.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Extract file bytes
            with zf.open(member, "r") as src_f, open(dest_path, "wb") as dst_f:
                dst_f.write(src_f.read())

def extract_zip(zip_path: Path, out_root: Path) -> bool:
    out_dir = out_root / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Quick integrity check
            bad = zf.testzip()
            if bad is not None:
                print(f"[WARN] {zip_path.name}: first bad file: {bad} (skipping)", file=sys.stderr)
                return False
            safe_extract(zf, out_dir)
        print(f"[OK]   Extracted: {zip_path.name} -> {out_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"[ERR]  Bad ZIP: {zip_path}", file=sys.stderr)
    except RuntimeError as e:
        print(f"[ERR]  {zip_path.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERR]  {zip_path.name}: unexpected error: {e}", file=sys.stderr)
    return False

def main():
    ap = argparse.ArgumentParser(description="Decompress all .zip files in a folder.")
    ap.add_argument("--src", required=True, help="Source directory containing ZIP files")
    ap.add_argument("--out", required=True, help="Output directory for extracted contents")
    ap.add_argument("--recursive", action="store_true", help="Search source directory recursively")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not src.exists() or not src.is_dir():
        print(f"[ERR] Source directory not found or not a directory: {src}", file=sys.stderr)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    zips = find_zip_files(src, args.recursive)
    if not zips:
        print(f"[INFO] No .zip files found in {src} (recursive={args.recursive}).")
        sys.exit(0)

    print(f"[INFO] Found {len(zips)} zip file(s). Extracting to: {out}\n")

    ok = 0
    for zp in zips:
        if extract_zip(zp, out):
            ok += 1

    print(f"\n[SUMMARY] Extracted {ok}/{len(zips)} archives successfully.")
    sys.exit(0 if ok == len(zips) else 2)

if __name__ == "__main__":
    main()