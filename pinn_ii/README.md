# -*- coding: utf-8 -*-
"""Potential field PINNs: J = 0"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import botf
import potf_fft


class PeriodicMLP(nn.Module):
    """
    Psi(x, y, z) for horizontally periodic potential field.
    Input x,y are encoded with sin/cos so horizontal periodicity is built in.
    """

    def __init__(
        self,
        lx: float,
        ly: float,
        lz: float,
        width: int = 128,
        depth: int = 5,
    ):
        super().__init__()
        self.lx = lx  # X方向の領域サイズ
        self.ly = ly  # Y方向の領域サイズ
        self.lz = lz  # Z方向の領域サイズ

        in_dim = 5  # sin x, cos x, sin y, cos y, z
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, z):
        # 周期境界に対応するため、座標を三角関数で入力
        xx = 2.0 * torch.pi * x / self.lx
        yy = 2.0 * torch.pi * y / self.ly

        # zは[-1, 1]に正規化して入力
        zz = 2.0 * z / self.lz - 1.0

        inp = torch.cat(
            [torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy), zz],
            dim=-1,
        )
        return self.net(inp)


def grad(u, x):
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian_cartesian(psi, x, y, z):
    psi_x = grad(psi, x)
    psi_y = grad(psi, y)
    psi_z = grad(psi, z)

    psi_xx = grad(psi_x, x)
    psi_yy = grad(psi_y, y)
    psi_zz = grad(psi_z, z)

    return psi_xx + psi_yy + psi_zz


def sample_periodic_b0(b0_grid, x, y, lx, ly):
    """
    Bilinear sampling from Bz bottom grid.

    b0_grid: tensor with shape (ny, nx)
    x, y: tensors with shape (n, 1), physical coordinates
    """

    ny, nx = b0_grid.shape

    # Periodic wrap
    xw = torch.remainder(x, lx)
    yw = torch.remainder(y, ly)

    # Convert to grid indices
    gx = xw / lx * nx
    gy = yw / ly * ny

    i0 = torch.floor(gx).long() % nx
    j0 = torch.floor(gy).long() % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny

    tx = gx - torch.floor(gx)
    ty = gy - torch.floor(gy)

    b00 = b0_grid[j0.squeeze(-1), i0.squeeze(-1)].unsqueeze(-1)
    b10 = b0_grid[j0.squeeze(-1), i1.squeeze(-1)].unsqueeze(-1)
    b01 = b0_grid[j1.squeeze(-1), i0.squeeze(-1)].unsqueeze(-1)
    b11 = b0_grid[j1.squeeze(-1), i1.squeeze(-1)].unsqueeze(-1)

    return (
        (1 - tx) * (1 - ty) * b00
        + tx * (1 - ty) * b10
        + (1 - tx) * ty * b01
        + tx * ty * b11
    )


def make_points(n, lx, ly, lz, device):
    x = lx * torch.rand(n, 1, device=device, requires_grad=True)
    y = ly * torch.rand(n, 1, device=device, requires_grad=True)
    z = lz * torch.rand(n, 1, device=device, requires_grad=True)
    return x, y, z


def train(
    b0_grid,
    lx,
    ly,
    lz,
    steps=20000,
    n_pde=4096,
    n_bc=2048,
    lr=1e-3,
    T_max=3000,
    eta_min=1e-5,
    device="cuda",
):
    b0_grid = torch.as_tensor(b0_grid, dtype=torch.float32, device=device)

    model = PeriodicMLP(lx=lx, ly=ly, lz=lz).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=T_max,
        eta_min=eta_min,
    )

    for step in range(steps):
        opt.zero_grad()

        # Interior PDE points
        x, y, z = make_points(n_pde, lx, ly, lz, device)
        psi = model(x, y, z)
        res = laplacian_cartesian(psi, x, y, z)
        loss_pde = torch.mean(res**2)

        # Bottom Neumann BC: Bz = -dPsi/dz = B0
        xb = lx * torch.rand(n_bc, 1, device=device, requires_grad=True)
        yb = ly * torch.rand(n_bc, 1, device=device, requires_grad=True)
        zb = torch.zeros(n_bc, 1, device=device, requires_grad=True)

        psi_b = model(xb, yb, zb)
        psi_z_b = grad(psi_b, zb)
        bz_b = -psi_z_b

        b0 = sample_periodic_b0(b0_grid, xb, yb, lx, ly)
        loss_bottom = torch.mean((bz_b - b0) ** 2)

        # Top condition: Psi = 0, equivalent to horizontal field vanishing up to gauge
        xt = lx * torch.rand(n_bc, 1, device=device, requires_grad=True)
        yt = ly * torch.rand(n_bc, 1, device=device, requires_grad=True)
        zt = lz * torch.ones(n_bc, 1, device=device, requires_grad=True)

        psi_t = model(xt, yt, zt)
        loss_top = torch.mean(psi_t**2)

        loss = loss_pde + 10.0 * loss_bottom + loss_top
        loss.backward()
        opt.step()
        scheduler.step()

        if step % 500 == 0:
            print(
                step,
                {
                    "loss": float(loss.detach().cpu()),
                    "pde": float(loss_pde.detach().cpu()),
                    "bottom": float(loss_bottom.detach().cpu()),
                    "top": float(loss_top.detach().cpu()),
                },
            )

    return model


@torch.no_grad()
def evaluate_field(model, x1d, y1d, z1d, device="cuda"):
    """
    Returns Bx, By, Bz on meshgrid with shape (nz, ny, nx).
    This function temporarily enables grad internally.
    """
    X, Y, Z = torch.meshgrid(
        torch.as_tensor(x1d, dtype=torch.float32, device=device),
        torch.as_tensor(y1d, dtype=torch.float32, device=device),
        torch.as_tensor(z1d, dtype=torch.float32, device=device),
        indexing="xy",
    )

    x = X.reshape(-1, 1).detach().requires_grad_(True)
    y = Y.reshape(-1, 1).detach().requires_grad_(True)
    z = Z.reshape(-1, 1).detach().requires_grad_(True)

    with torch.enable_grad():
        psi = model(x, y, z)
        bx = -grad(psi, x)
        by = -grad(psi, y)
        bz = -grad(psi, z)

    nx = len(x1d)
    ny = len(y1d)
    nz = len(z1d)

    bx = bx.reshape(ny, nx, nz).permute(2, 0, 1).detach().cpu()
    by = by.reshape(ny, nx, nz).permute(2, 0, 1).detach().cpu()
    bz = bz.reshape(ny, nx, nz).permute(2, 0, 1).detach().cpu()

    return bx, by, bz


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # domain
    lx, ly, x, y, b0 = botf.make_multimode_bottom_bz()

    # size of vertical domain
    lz = 0.5 * lx

    # train
    model = train(
        b0_grid=b0,
        lx=lx,
        ly=ly,
        lz=lz,
        steps=10000,
        n_pde=4096,
        n_bc=1024,
        lr=1.0e-3,
        T_max=3000,
        eta_min=1e-5,
        device=device,
    )

    # evaluate field
    z = np.linspace(0.0, lz, len(x))
    _, _, bz = evaluate_field(model, x, y, z, device=device)
    _, _, _, bz_ref = potf_fft.solve_potential_field_fft_cartesian_at_z(
        b0_grid=b0,
        lx=lx,
        ly=ly,
        lz=lz,
        z=z,
    )

    # ---- plot data ----#
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 8))

    # plot bottom Bz
    ax = axs[0, 0]
    im = ax.pcolormesh(y, x, bz[0].numpy())
    cb = fig.colorbar(im, location="top", pad=0.02, shrink=0.8)
    cb.set_label("Bz at z=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.label_outer()

    # plot top Bz
    ax = axs[0, 1]
    im = ax.pcolormesh(y, x, bz[-1].numpy())
    cb = fig.colorbar(im, location="top", pad=0.02, shrink=0.8)
    cb.set_label("Bz at z=Lz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.label_outer()

    # plot top Bz - Bz (ref)
    ax = axs[1, 0]
    im = ax.pcolormesh(y, x, (bz[-1] - bz_ref[-1]).numpy())
    cb = fig.colorbar(im, location="top", pad=0.02, shrink=0.8)
    cb.set_label("Bz (PINN) - Bz (FFT) at z=Lz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.label_outer()

    # plot top Bz (ref)
    ax = axs[1, 1]
    im = ax.pcolormesh(y, x, bz_ref[-1])
    cb = fig.colorbar(im, location="top", pad=0.02, shrink=0.8)
    cb.set_label("Bz (FFT) at z=Lz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.label_outer()

    fig_file = "potf_pinns.png"
    plt.savefig(fig_file, dpi=150)
    print(f"Saved: {fig_file}")

    # save model
    pt_file = "potf_pinns.pt"
    torch.save(model.state_dict(), pt_file)
    print(f"Saved: {pt_file}")


if __name__ == "__main__":
    main()
