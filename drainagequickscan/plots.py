import matplotlib.pyplot as plt
import numpy as np


class Plotting:

    def __init__(self, model):
        self.ml = model

    def get_xz_kwelsloot(self):
        """get x and z coordinates for plotting seepage ditch.

        Returns
        -------
        xs: np.array
            array containing x-coordinates of ditch
        zs: np.array
            array containing z-coordinates of ditch
        """

        breedte_bodem_sloot = self.ml.breedte_kwelsloot / 2.
        breedte_talud_sloot = breedte_bodem_sloot / 2.

        dz = self.ml.z_deklaag_boven - self.ml.z_deklaag_onder
        z_sloot_onder = self.ml.z_deklaag_boven - dz / 3.

        xs = np.cumsum([0, breedte_talud_sloot, breedte_bodem_sloot,
                        breedte_talud_sloot])
        zs = np.array([self.ml.z_deklaag_boven, z_sloot_onder,
                       z_sloot_onder, self.ml.z_deklaag_boven])

        return xs, zs

    def xsec(self, inhom_bounds=False, show_layers=False, ax=None, **kwargs):

        # add points far away left and right
        xd = np.r_[-(20 + self.ml.x0) * np.ones(1), self.ml.xpts_mv[0],
                   self.ml.xpts_mv, 1000 * np.ones(1)]
        # add elevation to points far away left and right
        zd = np.r_[self.ml.z_voorland_onder, self.ml.z_voorland_onder,
                   self.ml.zpts_mv, self.ml.z_deklaag_boven]

        if ax is None:
            fig, ax = plt.subplots(1, 1, **kwargs)
        else:
            fig = ax.figure

        # plot surface level
        ax.plot(xd, zd, "g-", lw=2)

        # color confining layer as gray
        ax.fill_between(xd, self.ml.z_deklaag_onder * np.ones(len(xd)), zd, color="lightgray", alpha=1.0)

        # add river
        hbw = ax.fill_between(xd[:5], np.array([self.ml.z_deklaag_onder, self.ml.z_deklaag_onder,
                                                self.ml.z_deklaag_onder, self.ml.z_voorland_boven,
                                                self.ml.z_voorland_boven]), self.ml.hriv * np.ones(5),
                              color="LightBlue")
        hbw.set_zorder(0)

        # add aquifer
        ax.fill_between(np.array([xd[0], 400]), self.ml.z_wvp_onder * np.ones(2),
                        self.ml.z_deklaag_onder * np.ones(2), color="gold", alpha=0.5)

        # layers
        if show_layers:
            for i in range(0, len(self.ml.z), 2):
                if self.ml.z[i] == self.ml.z[-1]:
                    break
                elif self.ml.z[i] == self.ml.z[i + 1]:
                    ax.axhline(self.ml.z[i], color="lightgray", lw=0.5)

        # bottom of model
        ax.plot([xd[0], 400], [self.ml.z_wvp_onder] * 2, color="DarkSlateGray", lw=5, ls="solid")

        # add sdl
        if self.ml.sdl:
            ax.fill_between(np.array([xd[0], 400]), self.ml.z_sdl_onder * np.ones(2),
                            self.ml.z_sdl_boven * np.ones(2), color="lightgray")

        if inhom_bounds:
            b = self.ml.get_inhom_bounds()
            for ib in b:
                ax.axvline(ib, 0, 1, linestyle="dashed", color="k", linewidth=1)

        # viewing extent
        ax.set_ylim(bottom=self.ml.z_wvp_onder, top=self.ml.z_dijk_kruin + 5)
        ax.set_xlim(-1 * self.ml.x0 - 20., 400)

        # axis labels
        ax.set_ylabel("m NAP")
        ax.set_xlabel("afstand vanaf binnenteen (m)")

        # # grid
        # ax.grid(b=True)

        return fig, ax

    def inhoms(self, ax=None, xlim=None, **kwargs):

        if xlim is None:
            xlim = [-(20 + self.ml.x0), 400.]

        if ax is None:
            fig, ax = plt.subplots(1, 1, **kwargs)
        else:
            fig = ax.figure

        for inh in self.ml.inhoms:
            xL = inh.x1 if np.isfinite(inh.x1) else np.amin([-(20 + self.ml.x0), xlim[0]])
            xR = inh.x2 if np.isfinite(inh.x2) else np.amax([400., xlim[1]])

            # semi-confining layer
            ax.fill_between([xL, xR], inh.z[0], inh.z[1], facecolor="gray")

            # layers and aquitards
            for i in range(0, len(inh.z), 2):
                if inh.z[i] == inh.z[-1]:
                    ax.fill_between([xL, xR], inh.z[i], inh.z[i] - 1.0, facecolor=(0.3, 0.3, 0.3))
                elif inh.z[i] == inh.z[i + 1]:
                    ax.hlines(inh.z[i], xL, xR, color="lightgray", lw=0.5)
                elif inh.z[i] > inh.z[i + 1]:
                    ax.fill_between([xL, xR], inh.z[i], inh.z[i + 1], facecolor="lightgray")

            # bottom of model
            ax.plot([xL, xR], [self.ml.z_wvp_onder] * 2, color="DarkSlateGray", lw=5, ls="solid")

            # hstar
            ax.plot([xL, xR], [inh.hstar] * 2, ls="solid", lw=0.75, c="b")

        # plot boundaries
        b = self.ml.get_inhom_bounds()
        for ib in b:
            ax.axvline(ib, 0, 1, linestyle="dashed", color="k", linewidth=1)

        # axis labels
        ax.set_ylabel("m NAP")
        ax.set_xlabel("x (m)")

        # viewing extent
        ax.set_ylim(bottom=self.ml.z_wvp_onder, top=self.ml.z_dijk_kruin)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        return fig, ax

    def topview(self, xlim=None, ylim=(-100, 100), ax=None, labels=True,
                inhom_bounds=False, add_wlvl=False):

        if xlim is None:
            xlim = [self.ml.xR_riv - 20, 100.]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(16 * 0.75, 5 * 0.75))

        ax.fill_between([xlim[0], self.ml.xL_voorland], ylim[0] * np.ones(2), ylim[-1] * np.ones(2),
                        color="CornFlowerBlue")
        ax.fill_between([self.ml.xL_voorland, -self.ml.breedte_dijk_totaal], ylim[0] * np.ones(2),
                        ylim[-1] * np.ones(2),
                        color="CornFlowerBlue", alpha=0.8)
        ax.fill_between([-self.ml.breedte_dijk_totaal, 0], ylim[0] * np.ones(2), ylim[-1] * np.ones(2),
                        color="DarkGreen", alpha=0.5)
        ax.fill_between([0, xlim[-1]], ylim[0] * np.ones(2), ylim[-1] * np.ones(2),
                        color="YellowGreen", alpha=0.5)

        if inhom_bounds:
            b = self.ml.get_inhom_bounds()
            for ib in b:
                ax.axvline(ib, 0, 1, linestyle="dashed", color="gray", linewidth=1)

        if labels:
            if add_wlvl:
                lbl1 = "oppervlaktewater \n(NAP{0:+.1f}m)".format(self.ml.inhoms[0].hstar)
                lbl2 = "voorland \n(NAP{0:+.1f}m)".format(self.ml.inhoms[1].hstar)
                lbl3 = "achterland \n(NAP{0:+.1f}m)".format(self.ml.inhoms[-1].hstar)
            else:
                lbl1 = "oppervlaktewater"
                lbl2 = "voorland"
                lbl3 = "achterland"

            ax.text((xlim[0] + self.ml.xL_voorland) / 2., 0, lbl1,
                    rotation=270, ha="center", va="center")
            ax.text((self.ml.xL_voorland - self.ml.breedte_dijk_totaal) / 2., 0, lbl2,
                    rotation=0, ha="center", va="center")
            ax.text((-self.ml.breedte_dijk_totaal) / 2., 0, "  dijk",
                    rotation=0, ha="center", va="center")
            ax.text(xlim[-1] / 2., 0, lbl3,
                    rotation=0, ha="center", va="center")

        ax.set_xlim(xlim[0], xlim[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("afstand langs dijk vanaf midden (m)")
        ax.set_xlabel("afstand vanaf binnenteen (m)")

        return ax.figure, ax

    def topview_nodike(self, xlim=None, ylim=(-100, 100), ax=None, labels=True,
                       inhom_bounds=False, add_wlvl=False):

        if xlim is None:
            xlim = [self.ml.xR_riv - 20, 100.]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(16 * 0.75, 5 * 0.75))
        else:
            fig = ax.figure

        ax.fill_between([xlim[0], self.ml.xL_voorland], ylim[0] * np.ones(2), ylim[-1] * np.ones(2),
                        color="CornFlowerBlue")
        ax.fill_between([self.ml.xL_voorland, self.ml.xR_voorland], ylim[0] * np.ones(2), ylim[-1] * np.ones(2),
                        color="Gray", alpha=0.8)
        ax.fill_between([self.ml.xR_voorland, xlim[-1]], ylim[0] * np.ones(2), ylim[-1] * np.ones(2),
                        color="LightGray", alpha=0.5)

        if inhom_bounds:
            b = self.ml.get_inhom_bounds()
            for ib in b:
                ax.axvline(ib, 0, 1, linestyle="dashed", color="gray", linewidth=1)

        if labels:
            if add_wlvl:
                lbl1 = "oppervlaktewater \n(NAP{0:+.1f}m)".format(self.ml.inhoms[0].hstar)
                lbl2 = "voorland \n(NAP{0:+.1f}m)".format(self.ml.inhoms[1].hstar)
                lbl3 = "achterland \n(NAP{0:+.1f}m)".format(self.ml.inhoms[-1].hstar)
            else:
                lbl1 = "oppervlaktewater"
                lbl2 = "voorland"
                lbl3 = "achterland"

            ax.text((xlim[0] + self.ml.xL_voorland) / 2., 0, lbl1,
                    rotation=270, ha="center", va="center")
            ax.text((self.ml.xL_voorland + self.ml.xR_voorland) / 2., 0, lbl2,
                    rotation=0, ha="center", va="center")
            ax.text((xlim[-1] + self.ml.xR_voorland) / 2., 0, lbl3,
                    rotation=0, ha="center", va="center")

        ax.set_xlim(xlim[0], xlim[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("afstand langs dijk vanaf midden (m)")
        ax.set_xlabel("afstand vanaf binnenteen (m)")

        return fig, ax
