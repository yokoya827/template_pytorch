"""FFTによるCartesianポテンシャル磁場ソルバ."""

import numpy as np


def solve_potential_field_fft_cartesian_at_z(
    b0_grid: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """水平周期Cartesian領域のポテンシャル磁場を任意のz位置で評価する.

    Parameters
    ----------
    b0_grid
        shape=(ny, nx) の下端垂直磁場 Bz(x, y, z=0)。
    lx, ly, lz
        計算領域サイズ。
    z
        shape=(nz,) の評価位置。0 <= z <= lz を想定する。

    Returns
    -------
    psi
        shape=(nz, ny, nx) のスカラー磁気ポテンシャル。
    bx, by, bz
        shape=(nz, ny, nx) の磁場成分。
    """
    b0_grid = np.asarray(b0_grid, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if z.ndim != 1:
        raise ValueError("z must be a one-dimensional array.")

    if np.any(z < 0.0) or np.any(z > lz):
        raise ValueError("z must satisfy 0 <= z <= lz.")

    ny, nx = b0_grid.shape
    nz = z.size

    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)

    b0_hat = np.fft.rfft2(b0_grid)

    nkx = nx // 2 + 1

    psi_hat = np.zeros((nz, ny, nkx), dtype=np.complex128)
    bx_hat = np.zeros_like(psi_hat)
    by_hat = np.zeros_like(psi_hat)
    bz_hat = np.zeros_like(psi_hat)

    for j in range(ny):
        kyj = ky[j]

        for i in range(nkx):
            kxi = kx[i]
            k = np.sqrt(kxi**2 + kyj**2)

            b0_mode = b0_hat[j, i]

            if k == 0.0:
                # Mean vertical flux mode:
                #
                # Psi = B0_mean * (Lz - z)
                # Bz  = B0_mean
                psi_mode = b0_mode * (lz - z)
                bz_mode = np.full_like(z, b0_mode, dtype=np.complex128)
            else:
                # Stable expressions:
                #
                # sinh(k(L-z)) / cosh(kL)
                # = [exp(-kz) - exp(-k(2L-z))] / [1 + exp(-2kL)]
                #
                # cosh(k(L-z)) / cosh(kL)
                # = [exp(-kz) + exp(-k(2L-z))] / [1 + exp(-2kL)]
                e1 = np.exp(-k * z)
                e2 = np.exp(-k * (2.0 * lz - z))
                den = 1.0 + np.exp(-2.0 * k * lz)

                sinh_ratio = (e1 - e2) / den
                cosh_ratio = (e1 + e2) / den

                psi_mode = (b0_mode / k) * sinh_ratio
                bz_mode = b0_mode * cosh_ratio

            psi_hat[:, j, i] = psi_mode

            bx_hat[:, j, i] = -1j * kxi * psi_mode
            by_hat[:, j, i] = -1j * kyj * psi_mode
            bz_hat[:, j, i] = bz_mode

    psi = np.fft.irfft2(psi_hat, s=(ny, nx), axes=(1, 2))
    bx = np.fft.irfft2(bx_hat, s=(ny, nx), axes=(1, 2))
    by = np.fft.irfft2(by_hat, s=(ny, nx), axes=(1, 2))
    bz = np.fft.irfft2(bz_hat, s=(ny, nx), axes=(1, 2))

    return psi, bx, by, bz


def main():
    import matplotlib.pyplot as plt

    from botf import make_multimode_bottom_bz

    lx, ly, y, x, bz_bottom = make_multimode_bottom_bz(nx=64, ny=128)

    lz = 0.5 * lx
    z = np.array([0.0, 0.1, 0.2, 0.5, 1.0]) * lz

    _, _, _, bz = solve_potential_field_fft_cartesian_at_z(
        b0_grid=bz_bottom,
        lx=lx,
        ly=ly,
        lz=lz,
        z=z,
    )

    # ---- plot data ----#
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))

    # plot bottom Bz
    ax = axs[0]
    im = ax.pcolormesh(y, x, bz[0])
    cb = fig.colorbar(im, location="top", pad=0.02, shrink=0.8)
    cb.set_label("Bz at z=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.label_outer()

    # plot top Bz
    ax = axs[1]
    im = ax.pcolormesh(y, x, bz[-1])
    cb = fig.colorbar(im, location="top", pad=0.02, shrink=0.8)
    cb.set_label("Bz at z=Lz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.label_outer()

    fig_file = "potf_fft.png"
    plt.savefig(fig_file, dpi=150)
    print(f"Saved: {fig_file}")


if __name__ == "__main__":
    main()
