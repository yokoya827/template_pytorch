"""下部境界条件の磁場サンプル"""

import numpy as np


def make_multimode_bottom_bz(
    nx: int = 64,
    ny: int = 64,
    lx: float = 2.0 * np.pi,
    ly: float = 2.0 * np.pi,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """平均ゼロの複数モード下端垂直磁場を作る.

    Returns
    -------
    lx, ly
        水平方向の領域サイズ
    x, y
        shape=(nx,) および shape=(ny,) の水平方向の座標配列
    b0
        shape=(ny, nx) の下端 Bz。
    """
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    bz_bottom = (
        np.sin(X) * np.cos(Y) + 0.5 * np.sin(2.0 * X + 0.3) * np.sin(Y)
    ).astype(np.float32)
    #単純な磁場を生成, 第一項:1周期の波. 第二:高周波で位相をずらす

    return lx, ly, x, y, bz_bottom
