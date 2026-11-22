# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

import hashlib
import json
from urllib.request import urlopen


def fetch_frames(manifest):
    directory = manifest.parent / "frames"
    directory.mkdir(exist_ok=True)
    for sample in json.loads(manifest.read_text()):
        path = directory / sample["file"]
        if path.exists():
            continue
        with urlopen(sample["url"], timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != sample["sha256"]:
            raise ValueError(f"Downloaded hash mismatch for {path}")
        pending = path.with_suffix(".download")
        pending.write_bytes(raw)
        pending.replace(path)
