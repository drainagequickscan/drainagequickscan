# %%
import matplotlib as mpl
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Lucida Sans Unicode']

mpl.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import pydijkdrainage as pdd

# %% Define parameters, see dict definitions stored in XsecDikeModel._xdict, etc.
xdict = {
    "breedte_voorland": 100.,
    "breedte_dijk_totaal": 60.,
    "breedte_dijk_kruin": 3.,
    "breedte_dijk_berm": 15.,
    "breedte_kwelsloot": 5.
}

zdict = {
    "z_dijk_kruin": 7.0,
    "z_dijk_berm": 4.0,
    "z_voorland_boven": 2,
    "z_voorland_onder": -1.,
    "z_deklaag_boven": 3.,
    "z_deklaag_onder": -1.,
    "z_sdl_boven": -10.,
    "z_sdl_onder": -13.,
    "z_wvp_onder": -20.
}

kdict = {
    "anisotropie": 0.25,
    "kv_deklaag_voorland": 0.01,
    "kv_deklaag_achterland": 0.01,
    "kv_sdl": 0.05,
    "kh_wvp": 15.
}

wdict = {
    "hriv": 6.5,
    "hpolder": 2.5,
    "criv": 1.0
}

# %% Initialize model and solve
xsm = pdd.XsecDikeModel(xdict, zdict, kdict, wdict)
xsm.solve()

# get heads
x = np.linspace(-(20+xsm.x0), 400, 1001)
y = np.zeros(len(x))
hds = xsm.model.headalongline(x, y)

# %% Horizontal Drainage
xsm_hd = pdd.HorizontalDrainXsec(pdd.XsecDikeModel(xdict, zdict, kdict, wdict), x_drn=0.0, drawdown=2.0)
xsm_hd.solve()

# heads
hds_hd = xsm_hd.model.headalongline(x, y)

# %% Plotting

# pretty cross-section
fig, ax = xsm.plots.xsec(inhom_bounds=True, show_layers=True, figsize=(10, 5))
xsm_hd.plot_hdrn(ax)
ax.plot(x, hds[0], lw=2, label="stijghoogte 1$^e$ wvp zonder drainagesysteem", c="C3", ls="dashed")
ax.plot(x, hds_hd[0], lw=2, label="stijghoogte 1$^e$ wvp met horizontale drain", c="C3", ls="solid")
ax.legend(loc="upper right")

# get streamlines
nx = 201  # no. of gridpts
xsm_hd.model.vcontoursf1D(-220, 400, nx, 30, labels=False, color="b", newfig=False, layout=False, ax=ax)

# label wvp
props = dict(boxstyle='round', facecolor='LightGoldenRodYellow', alpha=0.75)
zmid_wvp1 = np.mean([xsm.z_deklaag_onder, xsm.z_sdl_boven])
zmid_wvp2 = np.mean([xsm.z_wvp_onder, xsm.z_sdl_onder])
ax.text(385, zmid_wvp1, "wvp 1", fontsize=14,
        verticalalignment='center', ha="right", bbox=props)
ax.text(385, zmid_wvp2, "wvp 2", fontsize=14,
        verticalalignment='center', ha="right", bbox=props)

fig.savefig("./figures/xsec_stromingen_figure.png", dpi=300, bbox_inches="tight")

plt.show()
