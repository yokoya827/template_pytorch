import numpy as np

# 読み込み
data = np.load("/cidashome/sc/c0315yokota/template_pytorch/rtmag/data/train/b_0.150_0.040.npz")

# 中に入っている配列のキーを確認
print(data.files)

# 例: 'arr_0', 'arr_1' または保存時に付けた名前
for arr in data.values():
    print(arr)
