import logging

import numpy as np
import pandas as pd
import timml

from .plots import Plotting


# Custom exception for errors in input
class GeometryError(Exception):
    """Custom exception for geometry input errors.

    Parameters
    ----------
    Exception : GeometryError
        Geometry exception

    """

    def __init__(self, message, *errors):
        super(GeometryError, self).__init__(message)
        self.errors = errors


class XsecDikeModel:
    """Wrapper around TimML cross-sectional model"""

    _xdict = {
        "breedte_voorland": np.nan,
        "breedte_dijk_totaal": np.nan,
        "breedte_dijk_kruin": np.nan,
        "breedte_dijk_berm": np.nan,
        "breedte_kwelsloot": np.nan,
    }

    _zdict = {
        "z_dijk_kruin": np.nan,
        "z_dijk_berm": np.nan,
        "z_voorland_boven": np.nan,
        "z_voorland_onder": np.nan,
        "z_deklaag_boven": np.nan,
        "z_deklaag_onder": np.nan,
        "z_sdl_boven": np.nan,
        "z_sdl_onder": np.nan,
        "z_wvp_onder": np.nan,
    }

    _kdict = {
        "anisotropie": 1.0,
        "kv_deklaag_voorland": np.nan,
        "kv_deklaag_achterland": np.nan,
        "kv_sdl": np.nan,
        "kh_wvp": np.nan,
    }

    _wdict = {"hriv": np.nan, "hpolder": np.nan, "criv": 1.0}

    def __init__(self, xdict, zdict, kdict, wdict):
        self.logger = self.get_logger()

        # PARAMETERS
        self.init_time = pd.Timestamp.now()
        self.logger.info("Time start: " + self.init_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("-" * 15 + " SETTING UP MODEL " + "-" * 15)
        self.logger.info("Checking input parameters...")

        self.xdict = xdict
        self.zdict = zdict
        self.kdict = kdict
        self.wdict = wdict

        # test for completeness and add to object as attributes
        for idict, check_dict in zip(
            [xdict, zdict, kdict, wdict],
            [self._xdict, self._zdict, self._kdict, self._wdict],
        ):
            s_in = set(idict.keys())
            s_expected = set(check_dict.keys())
            for key in s_expected - s_in:
                self.logger.error(
                    "Data Missing Error: {} is missing from input!".format(key)
                )

            for k, v in idict.items():
                if k in check_dict.keys():
                    setattr(self, k, v)

        # parameter used to divide aquifer into sub-layers
        self.L_filter = 5.0

        # get coordinates of surface level
        self.xpts_mv, self.zpts_mv = self.get_xz_arrays()

        # set coordinates of inhoms for easy access
        self.xR_riv = -self.breedte_voorland - self.breedte_dijk_totaal
        self.xL_voorland = self.xR_riv
        self.xR_voorland = self.xpts_mv[2]  # zone voorland loopt tot aan kruin dijk
        self.xL_achterland = self.xR_voorland
        self.logger.info("... All requisite parameters are defined.")

        # create z, k and c arrays
        self.initialize()

        # check if input makes sense
        self.verify_input()

        # build timml model
        self.build_timml_model()

        # Add plotting functionality
        self.plots = Plotting(self)

    @classmethod
    def fromparamfile(cls, fparam):
        xdict = {}
        zdict = {}
        kdict = {}
        wdict = {}

        with open(fparam, "r") as f:
            for line in f.readlines():
                k = line.split(": ")[0]
                v = np.float(line.split(": ")[1])
                if k.startswith("breedte"):
                    xdict[k] = v
                elif k.startswith("z"):
                    zdict[k] = v
                elif k.startswith("k") or k.startswith("anisotropie"):
                    kdict[k] = v
                else:
                    wdict[k] = v

        return cls(xdict, zdict, kdict, wdict)

    def write_paramfile(self, output=None):
        if output is None:
            output = "./parameters.txt"
        with open(output, "w") as f:
            for idict in [self.xdict, self.zdict, self.kdict, self.wdict]:
                for k, v in idict.items():
                    f.write("{0}: {1}\n".format(k, v))
        self.logger.info("Parameter file saved to: {}".format(output))

    def solve(self, **kwargs):
        self.model.solve(**kwargs)
        self.logger.info(
            "Model solved. Time elapsed: {0:.1f} seconds".format(
                (pd.Timestamp.now() - self.init_time).total_seconds()
            )
        )

    def get_logger(self, log_level=logging.INFO, filename="info.log"):
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
        )
        fhandler = logging.FileHandler(filename=filename, mode="w")
        # fhandler.setFormatter('%(asctime)s | %(levelname)s : %(message)s')
        logger = logging.getLogger()
        logger.addHandler(fhandler)
        logger.setLevel(log_level)

        return logger

    def initialize(self):
        # Afgeleide parameters
        if np.isnan(self.kv_sdl):
            self.sdl = False
            self.c_sdl = None
        else:
            self.sdl = True
            self.c_sdl = (float(self.z_sdl_boven) - float(self.z_sdl_onder)) / float(
                self.kv_sdl
            )

        # Create layers
        dzl1 = 1.0  # m, min thickness layer

        if self.sdl:  # 2 watervoerende lagen
            self.logger.info(
                (
                    "Main aquifer contains aquitard between "
                    + "{0:+.1f} and {1:+.1f} mNAP!".format(
                        self.z_sdl_boven, self.z_sdl_onder
                    )
                    + " Resistance c = {0:.2f} days.".format(self.c_sdl)
                )
            )
            if self.z_deklaag_onder - self.L_filter == self.z_sdl_boven:
                zdis_tzl = np.arange(
                    self.z_deklaag_onder,
                    self.z_deklaag_onder - (self.L_filter + dzl1),
                    -dzl1,
                )
                self.logger.info(
                    "Vertical filters reach top of aquitard. "
                    + "Subdivide aquifer into {0} 1-meter layers.".format(
                        len(zdis_tzl) - 1
                    )
                )
            else:
                zdis1 = np.arange(
                    self.z_deklaag_onder,
                    self.z_deklaag_onder - (self.L_filter + dzl1),
                    -dzl1,
                )
                self.logger.info(
                    "Top aquifer is thicker than length vertical filters. Subdivide aquifer into "
                    + "{0} 1-meter layers to bottom of filter.".format(len(zdis1) - 1)
                )
                remaining_thickness = (
                    self.z_deklaag_onder - self.L_filter - self.z_sdl_boven
                )
                if remaining_thickness <= 10.0:
                    nlays = int(np.ceil(remaining_thickness / dzl1))
                    zdis2 = np.linspace(
                        self.z_deklaag_onder - (self.L_filter + dzl1),
                        self.z_sdl_boven,
                        nlays,
                    )
                    self.logger.info(
                        "Thickness top aquifer beneath vertical filter is <= 10 m. "
                        + "Subdivide into {0} layers.".format(len(zdis2) - 1)
                    )
                elif remaining_thickness > 10.0:
                    nlays = 4  # divide remaining thickness up in this many layers
                    self.logger.info(
                        "Thickness top aquifer beneath filter is > 10 m. "
                        + "Subdivide into {0} layers with linearly increasing thickness.".format(
                            nlays
                        )
                    )
                    dzlj = (
                        remaining_thickness
                        / np.sum(np.arange(1, nlays + 1))
                        * np.arange(1, nlays + 1)
                    )
                    zdis2 = self.z_deklaag_onder - (self.L_filter) * np.ones(nlays)
                    zdis2 -= np.cumsum(dzlj)
                zdis_tzl = np.concatenate([zdis1, zdis2])
            self.logger.info("Second aquifer modelled as single layer.")
            zdis_wvp = np.array([self.z_sdl_onder, self.z_wvp_onder])
            zdis = np.concatenate([zdis_tzl, zdis_wvp])

            ztdis_tzl = zdis_tzl[:-1]  # top of layers, tzl
            zbdis_tzl = zdis_tzl[1:]  # bottom of layers, tzl
            ztdis_wvp = np.array([self.z_sdl_onder])
            zbdis_wvp = np.array([self.z_wvp_onder])
            zt = np.concatenate([ztdis_tzl, ztdis_wvp])
            zb = np.concatenate([zbdis_tzl, zbdis_wvp])

            self.kh = self.kh_wvp * np.ones(len(zb))
            self.kv = self.anisotropie * self.kh
            c = [self.c_sdl]
        else:  # 1 watervoerende laag
            if self.z_deklaag_onder - self.L_filter == self.z_wvp_onder:
                zdis = np.arange(
                    self.z_deklaag_onder,
                    self.z_deklaag_onder - (self.L_filter + dzl1),
                    -dzl1,
                )
                self.logger.info(
                    "Vertical filters reach bottom of aquifer. "
                    + "Subdivide aquifer into {0} 1-meter layers.".format(len(zdis))
                )
            else:
                zdis1 = np.arange(
                    self.z_deklaag_onder,
                    self.z_deklaag_onder - (self.L_filter + dzl1),
                    -dzl1,
                )
                self.logger.info(
                    "Aquifer is thicker than length vertical filter. "
                    + "Subdivide aquifer into {0} 1-meter layers to bottom vertical filter.".format(
                        len(zdis1)
                    )
                )
                remaining_thickness = (
                    self.z_deklaag_onder - self.L_filter - self.z_wvp_onder
                )
                if remaining_thickness <= 10.0:
                    nlays = int(np.ceil(remaining_thickness / dzl1))
                    zdis2 = np.linspace(
                        self.z_deklaag_onder - (self.L_filter + dzl1),
                        self.z_wvp_onder,
                        nlays,
                    )
                    self.logger.info(
                        "Thickness aquifer beneath vertical filter is <= 10 m. "
                        + "Subdivide into {0} layers.".format(len(zdis2) - 1)
                    )
                elif remaining_thickness > 10.0:
                    nlays = 4  # divide remaining thickness up in this many layers
                    self.logger.info(
                        "Thickness aquifer beneath filter is > 10 m. "
                        + "Subdivide into {0} layers with linearly increasing thickness.".format(
                            nlays
                        )
                    )
                    dzlj = (
                        remaining_thickness
                        / np.sum(np.arange(1, nlays + 1))
                        * np.arange(1, nlays + 1)
                    )
                    zdis2 = self.z_deklaag_onder - self.L_filter * np.ones(nlays)
                    zdis2 -= np.cumsum(dzlj)

                zdis = np.concatenate([zdis1, zdis2])
            zt = zdis[:-1]
            zb = zdis[1:]

            self.kh = self.kh_wvp * np.ones(len(zb))
            self.kv = self.anisotropie * self.kh
            c = []

        self.c_deklaag = (
            self.z_deklaag_boven - self.z_deklaag_onder
        ) / self.kv_deklaag_achterland
        self.c_voorland = (
            self.z_voorland_boven - self.z_voorland_onder
        ) / self.kv_deklaag_voorland

        z = [(iz, jz) for iz, jz in zip(zt, zb)]
        z = np.array(z).ravel()

        self.zriv = np.r_[np.atleast_1d(self.z_voorland_onder), z]
        self.zvoorland = np.r_[np.atleast_1d(self.z_voorland_boven), z]
        self.zpolder = np.r_[np.atleast_1d(self.z_deklaag_boven), z]
        self.z = self.zpolder

        self.logger.info("Model has {} layers.".format(len(zb)))

        H = zt - zb
        Hleakylayer = zb[:-1] - zt[1:]

        self.caq = np.zeros(len(self.kh) - 1)
        for j in range(len(self.kh) - 1):
            self.caq[j] = H[j + 1] / (2 * self.kv[j + 1]) + H[j] / (2 * self.kv[j])
        self.caq[Hleakylayer > 1e-14] = np.asarray(c)

    def build_timml_model(self):
        ml = timml.ModelMaq(
            kaq=self.kh,
            z=self.z,
            c=np.r_[np.atleast_1d(self.c_deklaag), self.caq],
            npor=0.3,
            topboundary="semi",
            hstar=0.0,
        )
        inhom_riv = timml.StripInhomMaq(
            ml,
            x1=-np.inf,
            x2=self.xR_riv,
            kaq=self.kh,
            z=self.zriv,
            c=np.r_[np.atleast_1d(self.criv), self.caq],
            npor=0.3,
            topboundary="semi",
            hstar=self.hriv,
        )
        inhom_voorland = timml.StripInhomMaq(
            ml,
            x1=self.xR_riv,
            x2=self.xR_voorland,
            kaq=self.kh,
            z=self.zvoorland,
            c=np.r_[np.atleast_1d(self.c_voorland), self.caq],
            npor=0.3,
            topboundary="semi",
            hstar=self.hriv,
        )
        inhom_achterland = timml.StripInhomMaq(
            ml,
            x1=self.xR_voorland,
            x2=np.inf,
            kaq=self.kh,
            z=self.zpolder,
            c=np.r_[np.atleast_1d(self.c_deklaag), self.caq],
            npor=0.3,
            topboundary="semi",
            hstar=self.hpolder,
        )

        self.model = ml
        self.inhoms = [inhom_riv, inhom_voorland, inhom_achterland]

    def verify_input(self):
        # Check geometry
        if self.z_deklaag_boven <= self.z_deklaag_onder:
            raise GeometryError(
                "z_deklaag_boven ({0}) below z_deklaag_onder ({1})".format(
                    self.z_deklaag_boven, self.z_deklaag_onder
                )
            )

        if self.z_voorland_boven <= self.z_voorland_onder:
            raise GeometryError(
                "z_deklaag_boven ({0}) below z_deklaag_onder ({1})".format(
                    self.z_deklaag_boven, self.z_deklaag_onder
                )
            )

        if self.sdl:
            if not (
                self.z_wvp_onder
                < self.z_sdl_onder
                < self.z_sdl_boven
                < self.z_deklaag_onder
                < self.z_deklaag_boven
            ):
                raise GeometryError("layer elevations not consistent in polder.")
            if not (
                self.z_wvp_onder
                < self.z_sdl_onder
                < self.z_sdl_boven
                < self.z_voorland_onder
                < self.z_voorland_boven
            ):
                raise GeometryError("layer elevations not consistent in 'voorland'.")
        else:
            if not self.z_wvp_onder < self.z_deklaag_onder < self.z_deklaag_boven:
                raise GeometryError("layer elevations not consistent in polder.")
            if not self.z_wvp_onder < self.z_voorland_onder < self.z_voorland_boven:
                raise GeometryError("layer elevations not consistent in 'voorland'.")

        if self.z_voorland_onder != self.z_deklaag_onder:
            self.logger.warning(
                "Confining layer bottoms not equal: outside: {0:.1f}, inside: {1:.1f}".format(
                    self.z_voorland_onder, self.z_deklaag_onder
                )
            )
            self.logger.warning(
                "... Calculation is correct but output graphics show only inside value."
            )

        if not self.z_deklaag_boven < self.z_dijk_berm < self.z_dijk_kruin:
            self.logger.warning(
                "Elevation revetment (NAP{0:+.1f}) not between surface level (NAP{1:+.1f}) or top dike (NAP{2:+.1f})".format(
                    self.z_dijk_berm, self.z_deklaag_boven, self.z_dijk_kruin
                )
            )
            self.logger.warning(
                "... Setting revetment elevation equal to surface level (NAP{0:+.1f}).".format(
                    self.z_deklaag_boven
                )
            )
            self.z_dijk_berm = self.z_deklaag_boven

        checklevels = np.array([self.hriv, self.z_voorland_boven, self.z_deklaag_boven])
        check = self.z_dijk_kruin < checklevels
        if np.any(check):
            checkstr = ["peil_buitenwater", "z_voorland_boven", "z_deklaag_boven"]
            self.logger.error(
                "Elevation top dike (NAP{0:+.1f}) is lower than: {1}, {2}".format(
                    self.z_dijk_kruin, np.array(checkstr)[check], checklevels[check]
                )
            )
            raise GeometryError(
                "Elevation top dike (NAP{0:+.1f}) is lower than: {1}, {2}".format(
                    self.z_dijk_kruin, np.array(checkstr)[check], checklevels[check]
                )
            )

        if self.breedte_voorland <= 0:
            raise GeometryError("'Voorland' cannot have length <= 0.")

        if np.any(
            np.array(
                [
                    self.breedte_voorland,
                    self.breedte_dijk_berm,
                    self.breedte_dijk_totaal,
                    self.breedte_kwelsloot,
                ]
            )
            < 0.0
        ):
            raise GeometryError("One or more lengths is < 0.")

        if self.breedte_dijk_berm >= self.breedte_dijk_totaal:
            raise GeometryError("Revetment cannot be wider that width dike!")

        if self.hpolder <= self.z_deklaag_onder:
            self.logger.error(
                "Groundwater level in polder should not be below bottom confining layer!"
            )

        if self.sdl:
            kwaardes = np.array(
                [
                    self.kv_deklaag_voorland,
                    self.kv_deklaag_achterland,
                    self.kh_wvp,
                    self.kv_sdl,
                ]
            )
        else:
            kwaardes = np.array(
                [self.kv_deklaag_voorland, self.kv_deklaag_achterland, self.kh_wvp]
            )
        if np.any(kwaardes <= 0.0):
            raise ValueError(
                "One or more hydraulic conductivities less than or equal to 0."
            )

    def get_xz_arrays(self):
        """
        BRAD2, create dike profile. Input x and z parameters
        """

        self.x0 = self.breedte_voorland + self.breedte_dijk_totaal

        totale_breedte_taluds = (
            self.breedte_dijk_totaal - self.breedte_dijk_berm - self.breedte_dijk_kruin
        )

        dz_voorland_kruin = self.z_dijk_kruin - self.z_voorland_boven
        dz_kruin_berm = self.z_dijk_kruin - self.z_dijk_berm
        dz_berm_achterland = self.z_dijk_berm - self.z_deklaag_boven
        dz_helling_totaal = dz_voorland_kruin + dz_kruin_berm + dz_berm_achterland

        breedte_talud_buiten = (
            dz_voorland_kruin / dz_helling_totaal * totale_breedte_taluds
        )
        breedte_talud_binnen_kruin = (
            dz_kruin_berm / dz_helling_totaal * totale_breedte_taluds
        )
        breedte_talud_binnen_berm = (
            dz_berm_achterland / dz_helling_totaal * totale_breedte_taluds
        )

        x = np.array(
            [
                0.0,
                self.breedte_voorland,
                breedte_talud_buiten,
                self.breedte_dijk_kruin,
                breedte_talud_binnen_kruin,
                self.breedte_dijk_berm,
                breedte_talud_binnen_berm,
            ]
        )
        x = np.cumsum(x) - self.x0

        z = np.array(
            [
                self.z_voorland_boven,
                self.z_voorland_boven,
                self.z_dijk_kruin,
                self.z_dijk_kruin,
                self.z_dijk_berm,
                self.z_dijk_berm,
                self.z_deklaag_boven,
            ]
        )

        return x, z

    def get_inhom_bounds(self):
        bounds = []
        for ih in self.inhoms[:-1]:
            bounds.append(ih.x2)
        return np.array(bounds)

    def calc_required_drawdown(
        self, h_exit, rho_deklaag=15.0, rho_w=10.0, gamma_bu=1.3, gamma_up=1.3
    ):
        """Calculate required head in first aquifer to prevent heave.

        Parameters
        ----------
        h_exit : float
            phreatic water level at exit point
        rho_deklaag : float, optional
            volumetric weight of the semi-confining layer (the default is 15.)
        rho_w : float, optional
            volumetric weight of water (the default is 10.)
        gamma_bu : float, optional
            safety factor, value generally between 1 and 2 (the default is 1.3)
        gamma_up : float, optional
            safety factor for heave (the default is 1.3)

        """
        max_head = h_exit + (rho_deklaag - rho_w) * (
            self.z_deklaag_boven - self.z_deklaag_onder
        ) / (rho_w * gamma_up * gamma_bu)

        return max_head
