#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import qrcode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QR code for ZIP download URL")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/synthetic_phase_transition_package.zip",
        help="Download URL to encode into QR",
    )
    parser.add_argument("--output", default="web/zip_download_qr.png", help="Output PNG path")
    args = parser.parse_args()

    img = qrcode.make(args.url)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Saved QR to {out} for URL: {args.url}")


if __name__ == "__main__":
    main()
