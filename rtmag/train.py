import os
import json
import argparse
from pathlib import Path

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from neuralop.models import UNO

from rtmag.train.training import train, get_dataloaders

if __name__ == "__main__":


    #-----------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()#--configを取得
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as config:#configを開いて、辞書として読み取る
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    base_path = args.base_path#保存フォルダ作成
    os.makedirs(base_path, exist_ok=True)

    np.save(os.path.join(args.base_path, "args.npy"), args.__dict__)#使った引数や設定を該当フォルダに保存

    log_dir = Path(args.base_path) / "log"#logを保存するディレクトリ、/演算子はpathを結合する
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)#writer


    #-----------------------------------------------------------------------------------------
    if args.model["model_name"] == "UNO":
        model = UNO(
            hidden_channels = args.model["hidden_channels"],#モデル内部（中間層）の基本幅、学習モデルの層の幅
            in_channels = args.model["in_channels"],#zの入力するスライス数
            out_channels = args.model["out_channels"],#[b, 64, y, x, 3]、境界条件から予測しｚ＝64までの磁場ベクトルを出力
            lifting_channels = args.model["lifting_channels"],#入力（Z×C）を高次元に持ち上げる（lifting） ための埋め込みの幅
            projection_channels = args.model["projection_channels"],#高次元から出力空間へ写像（projection） する幅、出力する前の幅
            n_layers = args.model["n_layers"],#スペクトル演算ブロック（UNOブロック）の段数、モデルの深さ

            factorization = args.model["factorization"],#スペクトル重み（高次元テンソル）を低ランク分解する方式、テンソルを行列の式の席で表すことで必要なパラメータ数を減らす、Tucker 分解は、テンソルを「小さなコアテンソル」と「各モードの基底行列」の積に分解
            implementation = args.model["implementation"],#重みの扱い方、factorization設定はテンソルを分解して表現するよという意味
            rank = args.model["rank"],#低ランク分解のランク（または比率）

            uno_n_modes = args.model["uno_n_modes"], #各層がスペクトル（フーリエ）空間で保持するモード数
            uno_out_channels = args.model["uno_out_channels"],#各層の中間出力幅
            uno_scalings = args.model["uno_scalings"],#各層での空間スケーリング
        )
        #[b, out_channels, y, x, z]
    elif args.model["model_name"] == "UNO_dxdydz":
        class UNO_dxdydz(torch.nn.Module):#出力が5次元なのでclassで引数を再設定
            def __init__(self, args):#selfはモデルのインスタンス
                super().__init__()
                self.uno = UNO(
                    hidden_channels = args.model["hidden_channels"],
                    in_channels = args.model["in_channels"],
                    out_channels = args.model["out_channels"],
                    lifting_channels = args.model["lifting_channels"],
                    projection_channels = args.model["projection_channels"],
                    n_layers = args.model["n_layers"],
                    factorization = args.model["factorization"],
                    implementation = args.model["implementation"],
                    rank = args.model["rank"],

                    uno_n_modes = args.model["uno_n_modes"], 
                    uno_out_channels = args.model["uno_out_channels"],
                    uno_scalings = args.model["uno_scalings"],
                )
                self.fc = torch.nn.Linear(5, 3)#出力が5次元なので3次元に整形

            def forward(self, x):
                x = self.uno(x)#特徴抽出
                x = self.fc(x)#３次元に戻す、特徴抽出 → 物理量推定
                return x
        
        model = UNO_dxdydz(args)#モデルにパラメータを入力
    else:
        raise NotImplementedError

    optimizer = Adam(model.parameters(), lr=args.training['learning_rate'])#オプティマイザの準備、model.parameters() で学習可能なパラメータ（重み・バイアス）を渡す、lr で学習率を設定

    CHECKPOINT_PATH = os.path.join(args.base_path, "last.pt")#学習途中で保存したモデルを復元するためのパス設定

    if os.path.exists(CHECKPOINT_PATH):#保存済みモデルがあれば読み込む   
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        ck_epoch = checkpoint['epoch'] + 1
    else:
        ck_epoch = 0

    train_dataloader, val_dataloader = get_dataloaders(args)#configをロード

    train(model, optimizer, train_dataloader, val_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer)#学習する