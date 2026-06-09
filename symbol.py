import os
from pathlib import Path

src_dir = Path("/workspaces/NLFFF")
dst_dir = Path("/workspaces/template_pytorch/nlfff_links")

for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith(".nc"):
            target = Path(root) / f  # 元ファイル
            # NOAA番号は target.parts[-2] （例: "11078"）
            noaa_dir = target.parent.parent.name
            link_dir = dst_dir / noaa_dir
            link_dir.mkdir(parents=True, exist_ok=True)
            link_name = link_dir / f

            if not link_name.exists():
                os.symlink(target, link_name)
                print(f"Created: {link_name} -> {target}")
            else:
                print(f"Skipped: {link_name} (already exists)")
