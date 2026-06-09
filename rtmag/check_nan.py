import numpy as np
from pathlib import Path

def check_npz_for_nan(folder_path):
    name = []
    folder = Path(folder_path)
    npz_files = list(folder.glob("*.npz"))

    if not npz_files:
        print("No .npz files found in", folder)
        return

    for file in npz_files:
        has_nan = False
        with np.load(file) as data:
            for key in data.files:
                arr = data[key]
                if np.isnan(arr).any():
                    print(f"NaN detected in {file.name}, key={key}")
                    name.append(file.name)
                    has_nan = True
        if not has_nan:
            print(f"OK: {file.name}")
    print(name)

# 使用例
check_npz_for_nan("/workspaces/template_pytorch/rtmag/data/train")  # フォルダのパスを指定
