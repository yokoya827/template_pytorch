import os
import glob
import numpy as np

# 変換したいフォルダ
root_folder = "/workspaces/template_pytorch/rtmag-main/train_lowlou_data"

# 再帰的に npz ファイルを取得
npz_files = glob.glob(os.path.join(root_folder, "**", "*.npz"), recursive=True)

print(f"Found {len(npz_files)} npz files.")

for f in npz_files:
    try:
        data = np.load(f)
        # array1 → b に変換
        if "array1" in data.files:
            b = data["array1"]

            # 上書き保存（bとして）
            np.savez(f, b=b)
            print(f"Converted: {f}")
        else:
            print(f"Skipped (no array1): {f}")
    except Exception as e:
        print(f"Error processing {f}: {e}")
