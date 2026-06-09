import os
import random
import shutil

# --- 設定 ---
source_dir = "/workspaces/template_pytorch/rtmag-main/trained_data"  # ← 実在するパス
dest_root = "/workspaces/template_pytorch/rtmag-main/train_lowlou_data"
train_dir = os.path.join(dest_root, "train_5000")
val_dir = os.path.join(dest_root, "val_1000")
test_dir = os.path.join(dest_root, "test")

num_train = 5000
num_val = 1000
num_test = 1




# --- ファイル一覧を取得 ---
npz_files = [f for f in os.listdir(source_dir) if f.endswith(".npz")]
assert len(npz_files) >= num_train + num_val + num_test, "npzファイルが不足しています"

# --- ランダムにシャッフル ---
random.shuffle(npz_files)

# --- 分割 ---
train_files = npz_files[:num_train]
val_files = npz_files[num_train:num_train + num_val]
test_files = npz_files[num_train + num_val:num_train + num_val + num_test]

# --- コピー処理 ---
def copy_files(files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for f in files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(target_dir, f)
        shutil.copy2(src, dst)

print("Copying train files...")
copy_files(train_files, train_dir)

print("Copying val files...")
copy_files(val_files, val_dir)

print("Copying test files...")
copy_files(test_files, test_dir)

print("完了しました！")
