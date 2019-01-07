# %%
import matplotlib as mpl

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
x = np.linspace(-(20 + xsm.x0), 400, 1001)
y = np.zeros(len(x))
hds = xsm.model.headalongline(x, y)

# %% Plotting

# pretty cross-section
fig, ax = xsm.plots.xsec(inhom_bounds=True)
ax.plot(x, hds[0], lw=2, label="stijghoogte bovenzijde watervoerend pakket zonder drainagesysteem", c="C3")
ax.legend(loc="upper right")
# get streamlines
nx = 201  # no. of gridpts
xsm.model.vcontoursf1D(-220, 400, nx, 30, labels=False, color="b", newfig=False, layout=False, ax=ax)

# model cross-section
fig, ax = xsm.plots.inhoms()
ax.plot(x, hds[0], lw=2, label="stijghoogte bovenzijde watervoerend pakket zonder drainagesysteem", c="C3")
# get streamlines
xsm.model.vcontoursf1D(-220., 400, nx, 30, labels=False, color="b", newfig=False, layout=False, ax=ax)

# %% Horizontal Drainage
xsm_hd = pdd.HorizontalDrainXsec(pdd.XsecDikeModel(xdict, zdict, kdict, wdict), x_drn=20.5, drawdown=2.0)
xsm_hd.solve()

# heads
hds_hd = xsm_hd.model.headalongline(x, y)

# pretty cross-section
fig, ax = xsm_hd.plots.xsec(inhom_bounds=True, show_layers=True)
xsm_hd.plot_hdrn(ax)
ax.plot(x, hds_hd[0], lw=2, label="stijghoogte bovenzijde watervoerend met horizontale drains", c="C3")
ax.plot(x, hds[0], lw=2, ls="dashed", label="stijghoogte bovenzijde watervoerend pakket zonder drainagesysteem", c="C3")

ax.legend(loc="upper right")
# get streamlines
nx = 201  # no. of gridpts
xsm_hd.model.vcontoursf1D(-220, 400, nx, 30, labels=False, color="b", newfig=False, layout=False, ax=ax)

# %% Grindkoffer
xsm_gk = pdd.GrindkofferXsec(pdd.XsecDikeModel(xdict, zdict, kdict, wdict), x_gk=1.0, breedte_gk=2.0, drawdown=2.0)
xsm_gk.solve()

# heads
hds_gk = xsm_gk.model.headalongline(x, y)

# pretty cross-section
fig, ax = xsm_gk.plots.xsec(inhom_bounds=True, show_layers=True)
xsm_gk.plot_gk(ax, slope_gk=0.33)
ax.plot(x, hds_gk[0], lw=2, label="stijghoogte bovenzijde watervoerend pakket met grindkoffer", c="C3")
ax.plot(x, hds[0], lw=2, ls="dashed", label="stijghoogte bovenzijde watervoerend pakket zonder drainagesysteem", c="C3")

ax.legend(loc="upper right")
# get streamlines
nx = 201  # no. of gridpts
xsm_gk.model.vcontoursf1D(-220, 400, nx, 30, labels=False, color="b", newfig=False, layout=False, ax=ax)

fig, ax = xsm_gk.plots.inhoms(figsize=(12, 5))

fig.savefig(r"./figures/model_structure_grindkoffer.png", dpi=300, bbox_inches="tight")

# %% Verticale putten
xsm_vp = pdd.VerticalWellsXsec(pdd.XsecDikeModel(xdict, zdict, kdict, wdict),
                               xw=0, welldist=40., L_total=200., drawdown=2.0)
xsm_vp.model.solve()
xsm_vp.model2.solve()

# # hds check
# yy = np.linspace(-100, 100, 501)
# xx = np.zeros(len(yy))
# h1 = xsm_vp.model.head(0, 0, 0) * np.ones(len(yy))
# h2 = xsm_vp.model2.headalongline(xx, yy, 0)
# plt.figure()
# plt.plot(yy, h1)
# plt.plot(yy, h2[0])
# plt.plot(yy, h1+h2[0])

# solve model
xsm_vp.solve()

# hds check
yy = np.linspace(-100, 100, 501)
xx = np.zeros(len(yy))
h1 = xsm_vp.model.head(0, 0, 0) * np.ones(len(yy))
h2 = xsm_vp.model2.headalongline(xx, yy, 0)
plt.figure()
plt.plot(yy, h1)
plt.plot(yy, h2[0])
plt.plot(yy, h1 + h2[0])
plt.axhline(xsm_vp.hls, linestyle="dashed")

# heads
hds_vp = xsm_vp.headalongline(x, y)

# pretty cross-section
fig, ax = xsm_vp.plots.xsec(inhom_bounds=True, show_layers=True)
ax.plot(x, hds_vp[0], lw=2, label="stijghoogte bovenzijde watervoerend pakket met verticale putten", c="C3")
ax.plot(x, hds[0], lw=2, ls="dashed", label="stijghoogte bovenzijde watervoerend pakket zonder drainagesysteem", c="C3")
xsm_vp.plot_vp(ax)
ax.legend(loc="upper right")

# %%
# top view with contours
xg = np.r_[np.linspace(xsm_vp.xsm.xR_voorland - 200., xsm_vp.xsm.xR_voorland - 50., 21),
           np.linspace(xsm_vp.xsm.xR_voorland - 50., xsm_vp.xsm.xR_voorland + 75., 151)[1:],
           np.linspace(xsm_vp.xsm.xR_voorland + 75., xsm_vp.xsm.xR_voorland + 200., 21)[1:]]
yg = np.linspace(-100, 100, 101)
hgr = xsm_vp.headgrid(xg, yg, layers=[0])

# %%
fig, ax = xsm_vp.plots.topview(xlim=[-180, 200], ylim=[-100, 100], inhom_bounds=True, add_wlvl=True)
ax.plot(xsm_vp.xw, xsm_vp.yw, c="k", marker="o", ls="none", label="verticale bronnen")
levels = np.r_[np.arange(1.8, 3.0, 0.1), np.arange(3.0, 6.5, 0.5)]
cs = ax.contour(xg, yg, hgr[0], levels, cmap="RdYlBu", vmin=2.5)
plt.clabel(cs, fmt="%.1f m", inline=True, inline_spacing=5)
ax.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")
ax.legend(loc="upper left")
ax.set_ylim(-100, 100)

# %%
xg1 = np.linspace(xsm_vp.xsm.xR_voorland - 200., 200., 101)
yg1 = np.linspace(-100, 100, 11)
hgr1 = xsm_vp.model.headgrid(xg1, yg1, layers=[0])

xg2 = np.linspace(xsm_vp.xsm.xR_voorland - 50., 50., 101)
yg2 = np.linspace(-100, 100, 101)
hgr2 = xsm_vp.model2.headgrid(xg2, yg2, layers=[0])

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
_, ax0 = xsm_vp.plots.topview(xlim=[-180, 100], ylim=[-100, 100], ax=ax0, inhom_bounds=True, labels=False)
_, ax1 = xsm_vp.plots.topview(xlim=[-180, 100], ylim=[-100, 100], ax=ax1, inhom_bounds=True, add_wlvl=False)

levels = np.arange(2.8, wdict["hriv"] - 0.3, 0.3)
ax0.axvline(0., linestyle="dashed", color="Yellow", linewidth=3, label="oneindige lange \nonttrekking")
ax0.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")
cs0 = ax0.contour(xg1, yg1, hgr1[0], levels, cmap="RdBu")
plt.clabel(cs0, fmt="%.1f m", inline=True, inline_spacing=15)
ax0.legend(loc="upper left")
ax0.text((-180 + xsm_vp.xsm.xL_voorland) / 2., 0, "oppervlaktewater \n(NAP{0:+.1f}m)".format(xsm_vp.xsm.hriv),
         rotation=90, ha="center", va="center")
ax0.text(100 / 2., 0, "achterland \n(NAP{0:+.1f}m)".format(xsm_vp.xsm.hpolder),
         rotation=0, ha="center", va="center")
ax0.set_title("Doorsnede model met oneindige lange onttrekking")

# plot 2
levels = np.r_[np.arange(-1.0, 0.0, 0.05), np.arange(0.05, 0.2, 0.05)]
cs1 = ax1.contour(xg2, yg2, hgr2[0], levels, cmap="seismic_r", vmin=-0.5, vmax=0.5)
ax1.plot(xsm_vp.xw, xsm_vp.yw, c="k", marker=".", ls="none", label="verticale bronnen")
ax1.plot([], [], ls="dashed", c="gray", lw=1, label="grenzen tussen \nzones in model")
ax1.legend(loc="best")
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cs1.cmap)
sm.set_array([])
width = "45%"
height = 0.1
loc = "lower left"
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax1, width=width, height=height, loc=loc)
cb = plt.colorbar(sm, cax=cax, ax=ax, orientation="horizontal")
cax.xaxis.tick_top()
cax.xaxis.set_label_position("top")
cax.set_xlabel("stijghoogte (m t.o.v. ref)")

ax1.set_title("Quasi-3D model met verticale bronnen")
ax1.set_ylabel("")

fig.subplots_adjust(wspace=0.05)

# %% Test if save params and load params work:
# xsm.write_paramfile()
# xsm2 = XsecDikeModel.fromparamfile("./parameters.txt")

# %% Test flow checks etc.
for ixsm in [xsm_hd, xsm_gk, xsm_vp]:
    ixsm.check_velocity()
    ixsm.check_pump_needed()
    ixsm.check_drawdown()

plt.show()
