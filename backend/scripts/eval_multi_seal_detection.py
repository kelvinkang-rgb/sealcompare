"""
離線評估：多印鑑偵測

用法（在 backend container 內）：
  python backend/scripts/eval_multi_seal_detection.py --image-path /app/uploads/xxx.png --max-seals 160 --debug

也可以一次跑多張：
  python backend/scripts/eval_multi_seal_detection.py --image-path /app/uploads/a.png --image-path /app/uploads/b.png --debug

說明：
- 預設使用 SEAL_MULTI_METHOD=auto（可用環境變數覆蓋）。
- --debug 會輸出中間圖與 overlay 到 /app/logs/detect_multiple_seals/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# 讓 `python scripts/...` 也能正常 import `app.*`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.seal_detector import detect_multiple_seals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", action="append", required=True, help="要評估的圖片路徑（可重複指定多次）")
    parser.add_argument("--max-seals", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true", help="輸出 debug 產物到 /app/logs/detect_multiple_seals/")
    args = parser.parse_args()

    if args.debug:
        os.environ["SEAL_DETECTOR_DEBUG"] = "1"

    method = os.getenv("SEAL_MULTI_METHOD", "auto")
    print(f"[eval] SEAL_MULTI_METHOD={method} timeout={args.timeout} max_seals={args.max_seals} debug={bool(args.debug)}")

    for p in args.image_path:
        path = Path(p)
        if not path.exists():
            print(f"[eval] NOT_FOUND: {p}")
            continue
        r = detect_multiple_seals(str(path), timeout=args.timeout, max_seals=args.max_seals)
        print(f"[eval] {p} detected={r.get('detected')} count={r.get('count')} reason={r.get('reason')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


