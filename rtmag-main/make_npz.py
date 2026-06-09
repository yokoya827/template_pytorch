import numpy as np
import os
import glob
import torch
from rtmag.process.paper.load import MyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



path = "/workspaces/template_pytorch/rtmag-main/D:/models/lowlou_high_div/best_model.pt"
   

folder = "/workspaces/template_pytorch/rtmag/data/train"

# フォルダ内の .npz ファイル一覧を取得
npz_files = glob.glob(os.path.join(folder, "*.npz"))



for i in range(len(npz_files)):
    data = np.load(npz_files[i])
    b = data["b"]


    b = np.expand_dims(b, axis=0)
    b = np.transpose(b, (0, 3, 2, 1, 4))
    b = b[:, 0:1, :, :, :]
    model_input = b

    mm = MyModel(path, device=device)
    b = mm.get_pred_from_numpy(model_input)
    np.savez(f"/workspaces/template_pytorch/rtmag-main/trained_data/trained_data{i}.npz", b=b)