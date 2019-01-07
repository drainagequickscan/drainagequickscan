import numpy as np
import pandas as pd
import timml
from matplotlib.patches import Polygon
from scipy.integrate import quad
from timml.intlinesink import IntHeadDiffLineSink, IntFluxDiffLineSink


# Custom exception for errors in input
class NoDrainageNeededError(Exception):

    def __init__(self, message, *errors):
        super(NoDrainageNeededError, self).__init__(message)
        self.errors = errors


class HorizontalDrainXsec:

    def __init__(self, xsm, x_drn=0., drawdown=1.0, max_head=None):

        self.xsm = xsm
        self.logger = self.xsm.logger
        self.model = self.xsm.model
        self.x_drn = x_drn

        # just in case it wasn't already solved
        self.xsm.solve(silent=True)

        # determine drain elevation
        h0 = self.xsm.model.head(self.x_drn, 0., layers=[0])
        if max_head is not None:
            self.logger.info("Using given maximum head to determine drainage level.")
            self.hls = max_head
            if max_head >= h0:
                raise NoDrainageNeededError(
                    "Vereiste stijghoogte (NAP{0:+.2f} m) ligt hoger dan berekende stijghoogte ".format(max_head) +
                    "in huidige situatie (NAP{0:+.2f} m) bij locatie drainage!".format(h0[0]) +
                    " Er is dus geen drainage nodig!")
        else:
            self.hls = h0 - drawdown
            self.logger.info("Drainage level is calculated using given drawdown relative "
                             "to reference model. H_drn_max = {0:.2f}".format(self.hls[0]))

        # add horizontal drain
        self.add_horizontal_drn(self.x_drn, self.hls)

        # add plotting
        self.plots = self.xsm.plots

        self.solve = self.xsm.solve

    def add_horizontal_drn(self, x_drn, hdrn):
        self.ls = timml.HeadLineSink1D(self.xsm.model, xls=x_drn, hls=hdrn, label="horizontale drain")

    def plot_hdrn(self, ax):

        h, = ax.plot([self.x_drn, self.x_drn], [self.xsm.z_deklaag_onder - 1, self.xsm.z_deklaag_onder],
                     c="k", lw=4, label="horizontale drain", ls="solid")
        return h

    def seepage(self):
        Q = np.sum(self.model.disvec(self.xsm.xR_voorland, 0.))
        return Q

    def discharge(self):
        Q = np.sum(self.ls.parameters)
        return Q

    def velocity_at_drn(self, porosity=None, diameter=None):
        if porosity is None:
            porosity = 0.3
        if diameter is None:
            diameter = 0.45  # diameter boorgat horizontale drains

        return self.discharge() / (2 * np.pi * diameter / 2. * 1. * porosity)

    def check_velocity(self, diameter=None, porosity=None):
        if porosity is None:
            porosity = 0.3
        if diameter is None:
            diameter = 0.45

        vmax = 2 * self.xsm.kh_wvp  # Max snelheid op boorgatwand BodemEnergieNL
        v = self.velocity_at_drn(diameter=diameter, porosity=porosity)
        ratio = v / vmax

        msg = "Velocities: v, vmax, ratio: {0:.2f}, {1:.2f}, {2:.2f}".format(v, vmax, ratio)
        if ratio > 1.0:
            prefix = "Warning! "
        else:
            prefix = "OK! "
        self.logger.info(prefix + msg)
        return ratio

    def check_pump_needed(self):
        b = self.hls > self.xsm.hpolder
        if b:
            msg = "Vrije afwatering WEL mogelijk! H_drainage {0:.2f} > H_polder {1:.2f}"
        else:
            msg = "Vrije afwatering NIET mogelijk! H_drainage {0:.2f} < H_polder {1:.2f}"
        self.logger.info(msg.format(self.hls[0], self.xsm.hpolder))
        return b

    def check_drawdown(self):
        b = self.hls > self.xsm.z_deklaag_onder - 1.0
        if b:
            msg = "Vereiste stijghoogte {0:.1f}m boven onderkant drain!"
        else:
            msg = "Vereiste stijghoogte {0:.1f}m onder onderkant drain!"
        self.logger.info(msg.format(self.hls[0] - self.xsm.z_deklaag_onder - 1.0))
        return b


class GrindkofferXsec:

    def __init__(self, xsm, x_gk=0., breedte_gk=1.0, drawdown=1.0, max_head=None, c_gk=1e-3):
        self.xsm = xsm
        self.logger = self.xsm.logger
        self.model = self.xsm.model
        self.x_gk = x_gk
        self.breedte_gk = breedte_gk
        self.c_gk = c_gk

        # just in case it wasn't already solved
        self.xsm.solve(silent=True)

        # determine drain elevation
        h0 = self.xsm.model.head(self.x_gk, 0., layers=[0])
        if max_head is not None:
            self.logger.info("Using given maximum head to determine drainage level.")
            self.hls = max_head
            if max_head >= h0:
                raise NoDrainageNeededError(
                    "Vereiste stijghoogte (NAP{0:+.2f} m) ligt hoger dan berekende stijghoogte ".format(max_head) +
                    "in huidige situatie (NAP{0:+.2f} m) bij locatie drainage!".format(h0[0]) +
                    " Er is dus geen drainage nodig!")
        else:
            self.hls = h0 - drawdown
            self.logger.info("Drainage level is calculated using given drawdown relative "
                             "to reference model. H_drn_max = {0:.2f}".format(self.hls[0]))

        # add horizontal drain
        self.add_grindkoffer(self.x_gk, self.breedte_gk, self.hls)

        # add plotting
        self.plots = self.xsm.plots

        self.solve = self.xsm.solve

    def add_grindkoffer(self, x_gk, breedte_gk, hls):
        # edit polder inhom coordinates:
        inh_polder = self.xsm.inhoms[-1]
        inh_polder.x2 = x_gk

        # add new inhoms
        z_gk = self.xsm.zpolder.copy()
        z_gk[0] = z_gk[1]
        inhom_gk = timml.StripInhomMaq(self.model, x1=x_gk, x2=x_gk + breedte_gk, kaq=self.xsm.kh,
                                       z=z_gk, c=np.r_[np.atleast_1d(self.c_gk), self.xsm.caq],
                                       npor=0.3, topboundary="semi", hstar=hls)
        inh_polder2 = timml.StripInhomMaq(self.model, x1=x_gk + breedte_gk, x2=np.inf, kaq=self.xsm.kh,
                                          z=self.xsm.zpolder,
                                          c=np.r_[np.atleast_1d(self.xsm.c_deklaag), self.xsm.caq],
                                          npor=0.3, topboundary="semi", hstar=self.xsm.hpolder)
        # update inhoms list
        self.xsm.inhoms += [inhom_gk, inh_polder2]

    def plot_gk(self, ax, slope_gk=0.33):
        ddeklaag = self.xsm.z_deklaag_boven - self.xsm.z_deklaag_onder

        xgk = np.array([0., ddeklaag / slope_gk, ddeklaag / slope_gk + self.breedte_gk,
                        ddeklaag / slope_gk + self.breedte_gk + ddeklaag / slope_gk]) - ddeklaag / slope_gk + self.x_gk
        ygk = [self.xsm.z_deklaag_boven, self.xsm.z_deklaag_onder,
               self.xsm.z_deklaag_onder, self.xsm.z_deklaag_boven]
        xy = [(ix, iy) for ix, iy in zip(xgk, ygk)]
        p1 = Polygon(xy, fill=True, closed=True, hatch="oo", facecolor="DarkGray", edgecolor="black")
        ax.add_patch(p1)

    def seepage(self):
        Q = np.sum(self.model.disvec(self.xsm.xR_voorland, 0.))
        return Q

    def discharge(self):
        fqgk = lambda x: (self.model.headalongline(x, 0., layers=0) - self.hls) / self.c_gk
        Q = quad(fqgk, self.x_gk, self.x_gk + self.breedte_gk)
        return Q[0]

    def velocity_at_gk(self, porosity=None):

        if porosity is None:
            porosity = 0.3

        return self.discharge() / (self.breedte_gk * 1. * porosity)

    def check_velocity(self, porosity=None):
        if porosity is None:
            porosity = 0.3

        vmax = 2 * self.xsm.kh_wvp  # Max snelheid op boorgatwand BodemEnergieNL
        v = self.velocity_at_gk(porosity=porosity)
        ratio = v / vmax

        msg = "Velocities: v, vmax, ratio: {0:.2f}, {1:.2f}, {2:.2f}".format(v, vmax, ratio)
        if ratio > 1.0:
            prefix = "Warning! "
        else:
            prefix = "OK! "
        self.logger.info(prefix + msg)
        return ratio

    def check_pump_needed(self):
        b = self.hls > self.xsm.hpolder
        if b:
            msg = "Vrije afwatering WEL mogelijk! H_drainage {0:.2f} > H_polder {1:.2f}"
        else:
            msg = "Vrije afwatering NIET mogelijk! H_drainage {0:.2f} < H_polder {1:.2f}"
        self.logger.info(msg.format(self.hls[0], self.xsm.hpolder))
        return b

    def check_drawdown(self):
        b = self.hls > self.xsm.z_deklaag_onder
        if b:
            msg = "Vereiste stijghoogte {0:.1f}m boven onderzijde grindkoffer!"
        else:
            msg = "Vereiste stijghoogte {0:.1f}m onder onderzijde grindkoffer!"
        self.logger.info(msg.format(self.hls[0] - self.xsm.z_deklaag_onder))
        return b


class VerticalWellsXsec:

    def __init__(self, xsm, xw=0., drawdown=2.0, welldist=10., L_total=100., L_filter=5.0, max_head=None):

        self.xsm = xsm
        self.logger = self.xsm.logger
        self.model = self.xsm.model
        self.welldist = welldist
        self.L_total = L_total
        self.L_filter = L_filter

        self.nwells = int(L_total / welldist)
        self.rw = 0.25
        self.yw = np.linspace(-self.L_total / 2. + self.welldist / 2., self.L_total / 2. - self.welldist / 2.,
                              self.nwells)
        self.xw = xw * np.ones(len(self.yw))
        if self.nwells % 2 == 0:
            self.ycp = 0.
        else:
            self.ycp = np.mean(self.yw[int(np.floor(self.nwells / 2.)):int(np.ceil(self.nwells / 2.)) + 1])
        self.wlayers = np.where(self.xsm.model.aq.zaqbot >= self.xsm.z_deklaag_onder - self.L_filter)[0]

        # just in case it wasn't already solved
        self.xsm.solve(silent=True)

        # determine drain elevation
        h0 = self.xsm.model.head(0., 0., layers=[0])
        if max_head is not None:
            self.logger.info("Using given maximum head to determine drainage level.")
            self.hls = max_head
            if max_head >= h0:
                raise NoDrainageNeededError(
                    "Vereiste stijghoogte (NAP{0:+.2f} m) ligt hoger dan berekende stijghoogte ".format(max_head) +
                    "in huidige situatie (NAP{0:+.2f} m) bij locatie drainage!".format(h0[0]) +
                    " Er is dus geen drainage nodig!")
        else:
            self.hls = h0 - drawdown
            self.logger.info("Drainage level is calculated using given drawdown relative "
                             "to reference model. H_drn_max = {0:.2f}".format(self.hls[0]))

        # add infinite linesink to model 1 and solve:
        self.linesink = timml.HeadLineSink1D(self.xsm.model, xls=xw, hls=self.hls,
                                             label="horizontale drain", layers=self.wlayers)
        self.xsm.model.solve(silent=True)

        # initialize second model
        self.build_3D_timml_model(self.linesink)

        # add plotting
        self.plots = self.xsm.plots

    def solve(self, tol=0.05, silent=False):

        self.hmax = 99
        self.diff = 0.0
        print("Solving two models iteratively...")
        while self.hmax - self.hls > tol:
            # update first model drawdown
            self.xsm.model.remove_element(self.linesink)
            self.linesink = timml.HeadLineSink1D(self.xsm.model, xls=0.0, hls=self.hls + self.diff,
                                                 label="horizontale drain", layers=self.wlayers)
            # solve first model
            self.model.solve(silent=True)
            # update second model
            self.build_3D_timml_model(self.linesink)
            # solve second model
            self.model2.solve(silent=True)
            if not silent:
                print(".", end="", flush=True)
            # calculate remaining difference between target and current head
            self.hmax = self.model2.head(self.xw[0], self.ycp, layers=[0]) + self.model.head(0, 0, layers=[0])
            self.diff += self.hls - self.hmax
            msg = r"hmax={0:.2f}, hls={1:.2f}: Offset hls by {2:.3f}".format(self.hmax[0], self.hls[0], self.diff[0])
            self.logger.debug(msg)
        print()
        print("solution complete")
        self.logger.info("Time elapsed: {0:.1f} seconds".format((pd.datetime.now() -
                                                                 self.xsm.init_time).total_seconds()))

    def build_3D_timml_model(self, ls):
        # MODEL 2: Wells
        mlvp = timml.ModelMaq(kaq=self.xsm.kh, z=self.xsm.z,
                              c=np.r_[np.atleast_1d(self.xsm.c_deklaag), self.xsm.caq],
                              npor=0.3, topboundary="semi", hstar=0.)

        inhom_riv_vp = timml.StripInhomMaq(mlvp, x1=-np.inf, x2=self.xsm.xR_riv, kaq=self.xsm.kh,
                                           z=self.xsm.z, c=np.r_[np.atleast_1d(self.xsm.criv), self.xsm.caq],
                                           npor=0.3, topboundary="semi", hstar=0.)
        inhom_voorland_vp = timml.StripInhomMaq(mlvp, x1=self.xsm.xR_riv, x2=self.xsm.xR_voorland,
                                                kaq=self.xsm.kh, z=self.xsm.z, npor=0.3,
                                                c=np.r_[np.atleast_1d(self.xsm.c_voorland), self.xsm.caq],
                                                topboundary="semi", hstar=0.)
        inhom_achterland_vp = timml.StripInhomMaq(mlvp, x1=self.xsm.xR_voorland, x2=np.inf, kaq=self.xsm.kh,
                                                  z=self.xsm.z, topboundary="semi", npor=0.3,
                                                  c=np.r_[np.atleast_1d(self.xsm.c_deklaag), self.xsm.caq],
                                                  hstar=0.)

        inhom_riv_vp.addlinesinks = False
        inhom_voorland_vp.addlinesinks = False
        inhom_achterland_vp.addlinesinks = False

        maxlab = np.amax(self.xsm.model.aq.lab)
        ylim = self.L_total / 2. + 3 * maxlab
        dymin = np.amin([200., maxlab])
        rest = ylim - 3 * maxlab
        ypts = np.cumsum(np.r_[dymin * np.ones(int((np.ceil(rest / dymin)))),
                               np.array([maxlab / 2., maxlab / 2., maxlab, maxlab])])
        y = np.r_[-1 * ypts[::-1], np.zeros(1), ypts]

        self.model1d_with_2d_elements(mlvp, [inhom_riv_vp, inhom_voorland_vp, inhom_achterland_vp],
                                      y, order=3, ndeg=3)

        for i in self.wlayers:
            timml.LineSinkBase(mlvp, x1=self.xw[0], y1=-self.L_total / 2., x2=self.xw[0], y2=self.L_total / 2.,
                               Qls=-1 * ls.parameters[i] * self.L_total, layers=[i])

        self.Qwell = np.sum(ls.parameters * self.L_total) / self.nwells
        self.logger.debug("Qwell = {}".format(self.Qwell))
        self.well_list = []

        for iw in range(self.nwells):
            self.well_list.append(timml.Well(mlvp, xw=self.xw[iw], yw=self.yw[iw], Qw=self.Qwell, rw=self.rw, res=0.0,
                                             layers=self.wlayers))

        self.model2 = mlvp

    def model1d_with_2d_elements(self, ml, inhomlist, ypts, ndeg=3, order=5):

        lslist = []
        for i in range(len(inhomlist) - 1):
            inhomleft = inhomlist[i]
            inhomright = inhomlist[i + 1]
            for j in range(len(ypts) - 1):
                lsh = IntHeadDiffLineSink(ml, x1=inhomleft.x2, y1=ypts[j], x2=inhomleft.x2, y2=ypts[j + 1], order=order,
                                          ndeg=ndeg,
                                          addtomodel=True, aq=inhomright, aqin=inhomright, aqout=inhomleft)
                lsh.inhomelement = False

                lsq = IntFluxDiffLineSink(ml, x1=inhomright.x1, y1=ypts[j], x2=inhomright.x1, y2=ypts[j + 1],
                                          order=order,
                                          ndeg=ndeg,
                                          addtomodel=True, aq=inhomleft, aqin=inhomleft, aqout=inhomright)
                lsq.inhomelement = False
                lslist += [lsh, lsq]

        return lslist

    def headalongline(self, x, y, layers=None):
        h1 = self.model.headalongline(x, y, layers)
        h2 = self.model2.headalongline(x, y, layers)
        return h1 + h2

    def headgrid(self, xg, yg, layers=[0]):

        h1 = self.model.headalongline(xg, np.zeros(len(xg)), layers=layers)
        hgr1 = np.tile(h1[:, np.newaxis, :], (1, len(yg), 1))
        hgr2 = self.model2.headgrid(xg, yg, layers)

        return hgr1 + hgr2

    def plot_vp(self, ax):
        ax.plot([0, 0], [self.xsm.z_deklaag_onder - self.L_filter, self.xsm.z_deklaag_onder],
                c="k", lw=4, label="verticale put", ls="dotted")

    def seepage(self):
        fqvp = lambda y: np.sum(self.model.disvec(self.xsm.xR_voorland, y) +
                                self.model2.disvec(self.xsm.xR_voorland, y))
        fn = self.L_total / self.welldist
        Q = quad(fqvp, -self.welldist / 2., self.welldist / 2.)[0] * fn
        return Q

    def discharge(self):
        Q = self.Qwell * self.nwells
        return Q

    def headinwells(self):
        """ Get heads inside wells for each well

        Returns
        -------
        np.array
            head inside well for top-most screened layer for each well

        """
        return np.array([i.headinside()[0] + self.model.head(self.xw[0], 0, 0) for i in self.well_list])

    def velocity_at_well(self, rw=None, porosity=None):
        if porosity is None:
            porosity = 0.3
        if rw is None:
            rw = self.rw
        return self.discharge() / (2 * np.pi * rw / 2. * self.L_filter * porosity)

    def check_velocity(self, rw=None, porosity=None):
        if porosity is None:
            porosity = 0.3
        if rw is None:
            rw = self.rw
        # alpha = 6.  # factor voor permanente onttrekkingen, Grondwaterzakboekje 2016 p. 274
        # Q_max_vp = alpha * self.xsm.kh_wvp * self.L_filter * self.rw
        vmax = 2 * self.xsm.kh_wvp  # Max snelheid op boorgatwand BodemEnergieNL
        v = self.velocity_at_well(rw=rw, porosity=porosity)
        ratio = v / vmax
        msg = "Velocities: v, vmax, ratio: {0:.2f}, {1:.2f}, {2:.2f}".format(v, vmax, ratio)
        if ratio > 1.0:
            prefix = "Warning! "
        else:
            prefix = "OK! "
        self.logger.info(prefix + msg)
        return ratio

    def check_pump_needed(self):
        b = self.headalongline([self.xw[0] - self.rw], [self.yw[0]], layers=0)[0, 0] > self.xsm.hpolder
        if b:
            msg = "Vrije afwatering WEL mogelijk! H_drainage {0:.2f} > H_polder {1:.2f}"
        else:
            msg = "Vrije afwatering NIET mogelijk! H_drainage {0:.2f} < H_polder {1:.2f}"
        self.logger.info(msg.format(self.headalongline([self.xw[0] - self.rw], [self.yw[0]], layers=0)[0, 0],
                                    self.xsm.hpolder))
        return b

    def check_drawdown(self):
        b = self.headalongline([self.xw[0] - self.rw], [self.yw[0]], layers=0)[0, 0] > self.xsm.z_deklaag_onder - \
            self.L_filter
        if b:
            msg = "OK! Vereiste stijghoogte {0:.1f}m boven onderkant filter!"
        else:
            msg = "Error! Vereiste stijghoogte {0:.1f}m onder onderkant filter!"
        self.logger.info(msg.format(self.headalongline([self.xw[0] - self.rw], [self.yw[0]], layers=0)[0, 0] -
                                    (self.xsm.z_deklaag_onder - self.L_filter)))
        return b
