# %% Maak figuren voor Stromingen artikel
# (verticale bronnen bovenaanzichten)
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(1, "..")
import drainagequickscan as pdd

mpl.interactive(True)

# %% Define parameters, see dict definitions stored in XsecDikeModel._xdict, etc.
xdict = {
    "breedte_voorland": 100.0,
    "breedte_dijk_totaal": 60.0,
    "breedte_dijk_kruin": 3.0,
    "breedte_dijk_berm": 15.0,
    "breedte_kwelsloot": 5.0,
}

zdict = {
    "z_dijk_kruin": 7.0,
    "z_dijk_berm": 4.0,
    "z_voorland_boven": 2,
    "z_voorland_onder": -1.0,
    "z_deklaag_boven": 3.0,
    "z_deklaag_onder": -1.0,
    "z_sdl_boven": -10.0,
    "z_sdl_onder": -13.0,
    "z_wvp_onder": -20.0,
}

kdict = {
    "anisotropie": 0.25,
    "kv_deklaag_voorland": 0.01,
    "kv_deklaag_achterland": 0.01,
    "kv_sdl": 0.05,
    "kh_wvp": 15.0,
}

wdict = {"hriv": 6.5, "hpolder": 2.5, "criv": 1.0}

# %% Initialize model and solve
xsm = pdd.XsecDikeModel(xdict, zdict, kdict, wdict)
xsm.solve()

# %% Verticale putten
xsm_vp = pdd.VerticalWellsXsec(
    pdd.XsecDikeModel(xdict, zdict, kdict, wdict),
    xw=0,
    welldist=40.0,
    L_total=200.0,
    drawdown=2.0,
)
# solve model
xsm_vp.solve()

# %% Calculate grid for final result

xg = np.r_[
    np.linspace(xsm_vp.xsm.xR_voorland - 200.0, xsm_vp.xsm.xR_voorland - 50.0, 21),
    np.linspace(xsm_vp.xsm.xR_voorland - 50.0, xsm_vp.xsm.xR_voorland + 75.0, 151)[1:],
    np.linspace(xsm_vp.xsm.xR_voorland + 75.0, xsm_vp.xsm.xR_voorland + 100.0, 21)[1:],
]
yg = np.linspace(-45, 45, 101)
hgr = xsm_vp.headgrid(xg, yg, layers=[0])

# %% Top view with contours final result
fig, ax = xsm_vp.plots.topview(
    xlim=[-180, 100], ylim=[-100, 100], inhom_bounds=True, add_wlvl=True
)
ax.plot(xsm_vp.xw, xsm_vp.yw, c="k", marker="o", ls="none", label="verticale bronnen")
levels = np.r_[np.arange(1.8, 3.0, 0.1), np.arange(3.0, 6.5, 0.5)]
cs = ax.contour(xg, yg, hgr[0], levels, cmap="RdYlBu", vmin=2.5)
plt.clabel(cs, fmt="%.1f m", inline=True, inline_spacing=5)
ax.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")
ax.legend(loc="upper left")
ax.set_ylim(-45, 45)

# fig.savefig(r"./figures/verticale_bronnen_eindresultaat.png", dpi=300, bbox_inches="tight")

# %% Get headgrids for model 1 and model 2
xg1 = np.linspace(xsm_vp.xsm.xR_voorland - 200.0, 100.0, 101)
yg1 = np.linspace(-45, 45, 11)
hgr1 = xsm_vp.model.headgrid(xg1, yg1, layers=[0])

xg2 = np.linspace(xsm_vp.xsm.xR_voorland - 50.0, 50.0, 101)
yg2 = np.linspace(-45, 45, 101)
hgr2 = xsm_vp.model2.headgrid(xg2, yg2, layers=[0])

# %% Plot headgrid for model 1 and model 2
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
fig = plt.figure(figsize=(12, 10))
ax0 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax1 = plt.subplot2grid((3, 1), (1, 0), colspan=1, sharey=ax0, sharex=ax0)
ax2 = plt.subplot2grid((3, 1), (2, 0), colspan=1, sharey=ax0, sharex=ax0)

_, ax0 = xsm_vp.plots.topview_nodike(
    xlim=[-180, 100], ylim=[-45, 45], ax=ax0, inhom_bounds=True, labels=False
)
_, ax1 = xsm_vp.plots.topview_nodike(
    xlim=[-180, 100], ylim=[-45, 45], ax=ax1, inhom_bounds=True, labels=False
)
_, ax2 = xsm_vp.plots.topview_nodike(
    xlim=[-180, 100], ylim=[-45, 45], ax=ax2, inhom_bounds=True, add_wlvl=True
)

# Plot 3
p2 = ax1.axvline(
    0.0, linestyle="dashed", color="green", linewidth=4, label="lijn injectie"
)
(p1,) = ax2.plot(
    xsm_vp.xw,
    xsm_vp.yw,
    c="k",
    marker="o",
    ls="none",
    label="verticale bronnen",
    markersize=8,
)
levels = np.r_[np.arange(1.8, 3.0, 0.1), np.arange(3.0, 6.5, 0.5)]
cs = ax2.contour(
    xg, yg, hgr[0], levels, cmap="RdYlBu", vmin=2.5, vmax=6.2, linewidths=2
)
plt.clabel(cs, fmt="%.1f m", inline=True, inline_spacing=-10)
# ax2.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")


# Plot 1
p0 = ax0.axvline(
    0.0,
    linestyle="dashed",
    color="green",
    linewidth=4,
    label="oneindige lange \nonttrekking",
)
# ax0.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")

# contours
levels1 = np.arange(2.8, wdict["hriv"] - 0.3, 0.3)
cs0 = ax0.contour(
    xg1, yg1, hgr1[0], levels1, cmap="RdYlBu", linewidths=2, vmin=2.5, vmax=6.2
)
plt.clabel(cs0, fmt="%.1f m", inline=True, inline_spacing=-10)

# custom labels for zones
ax0.text(
    (-180 + xsm_vp.xsm.xL_voorland) / 2.0,
    0,
    "oppervlaktewater \n(NAP{0:+.1f}m)".format(xsm_vp.xsm.hriv),
    rotation=270,
    ha="center",
    va="center",
)
ax0.text(
    (xsm_vp.xsm.xR_voorland + 100) / 2.0,
    0,
    "achterland \n(NAP{0:+.1f}m)".format(xsm_vp.xsm.hpolder),
    rotation=0,
    ha="center",
    va="center",
)

# Plot 2
ax1.plot(
    xsm_vp.xw,
    xsm_vp.yw,
    c="k",
    marker=".",
    ls="none",
    label="verticale bronnen",
    zorder=11,
)
# ax1.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")

# custom labels for zones
ax1.text(
    (-180 + xsm_vp.xsm.xL_voorland) / 2.0,
    0,
    "oppervlaktewater \n(NAP{0:+.1f}m)".format(0.0),
    rotation=270,
    ha="center",
    va="center",
)
ax1.text(
    (xsm_vp.xsm.xR_voorland + xsm_vp.xsm.xL_voorland) / 2.0,
    0,
    "voorland \n(NAP{0:+.1f}m)".format(0.0),
    rotation=0,
    ha="center",
    va="center",
)
ax1.text(
    (xsm_vp.xsm.xR_voorland + 100) / 2.0,
    0,
    "achterland \n(NAP{0:+.1f}m)".format(0.0),
    rotation=0,
    ha="center",
    va="center",
)

# contours
levels2 = np.r_[np.arange(-1.0, 0.0, 0.05), np.arange(0.05, 0.2, 0.05)]
cs1 = ax1.contour(
    xg2,
    yg2,
    hgr2[0],
    levels2,
    cmap="seismic_r",
    vmin=-0.5,
    vmax=0.5,
    zorder=10,
    linewidths=2,
)

# Colorbar
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cs1.cmap)
sm.set_array([])
width = "25%"
height = 0.1
loc = "lower right"
cax = inset_axes(ax1, width=width, height=height, loc=loc)
cb = plt.colorbar(sm, cax=cax, ax=ax, orientation="horizontal")
cax.xaxis.tick_top()
cax.xaxis.set_label_position("top")
cax.set_xlabel("stijghoogte (m t.o.v. ref)")

# add dike
for iax in [ax0, ax1, ax2]:
    # iax.axvline(0.0, linestyle="dashdot", color="k", lw=1, label="zone dijk")
    # iax.axvline(-xsm_vp.xsm.breedte_dijk_totaal, linestyle="dashdot", color="k", lw=1)
    iax.axvspan(
        -xsm_vp.xsm.breedte_dijk_totaal,
        0.0,
        hatch="\/",
        color="k",
        label="dijk",
        fill=False,
        linestyle="dashed",
        zorder=1,
        alpha=0.45,
    )

p = mpl.patches.Patch(
    color="k", alpha=0.25, hatch=r"\\\///", linestyle="dashed", fill=False
)

# Set title/label/legend
ax0.set_title("Model 1: doorsnede met oneindig lange onttrekking", fontsize=12)
ax0.set_xlabel("")
ax0.legend([p0, p], ["oneindige lange \nonttrekking", "zone dijk"], loc="upper right")

ax1.set_title("Model 2: quasi-3D model met verticale bronnen", fontsize=12)
ax1.set_xlabel("")
ax1.legend(
    [p1, p2, p], ["verticale bronnen", "lijn injectie", "zone dijk"], loc="upper right"
)

ax2.set_title("Resultaat: stijghoogte 1$^e$ wvp van model 1 + model 2")
ax2.legend([p1, p], ["verticale bronnen", "zone dijk"], loc="upper right")
ax2.set_ylim(-45, 45)

# set whitespace
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.tight_layout()

fig.savefig(r"./figures/verticale_bronnen_uitleg2.png", dpi=300, bbox_inches="tight")
# fig.savefig(r"./figures/verticale_bronnen_uitleg2.svg", dpi=300, bbox_inches="tight")
