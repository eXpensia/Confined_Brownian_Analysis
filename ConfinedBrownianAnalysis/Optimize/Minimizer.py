import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Minimizer:
    """
    Class to manage the computation and optimization of the minizer using all the different observable.

    You can choose to compute the minimizers that you want by changing the to_compute variable.
    You can choose the variable to fit by changing the to_vary variable.

    Two different method are implemented to compute the minizer, sum all the chi_2 are
    average them. The method can be choose by changing the method variable.
    Note that all method name can be ofund in self.all_method variable.
    A new method can be implemented by adding the call inside the self.set_method method.
    """

    def __init__(
        self,
        Model,
        analysis,
        range_MSD_short=(1e-2, 1),
        range_plateau_MSD=(1e-2, 1e-1),
        range_plateau_C4=(1e-2, 1e-1),
        range_C4_short=(1e-2, 1e-1),
        range_diffusion=None,
        range_Peq=None,
        range_Feq=None,
        to_vary=["B", "lD", "Drho", "z_0", "b"],
        to_compute="all",
        method="sum",
    ):

        self.parameters = [
            "B",
            "lD",
        ]
        self.all_method = ["sum", "mean"]

        self.all_variable = ["B", "lD", "Drho", "z_0", "b", "a", "noise_lvl_MSD", "eta"]

        self.available_minimizer = [
            "minimizer_MSD_short_time",
            "minimizer_plateau_MSD",
            "minimizer_C4_short_time",
            "minimizer_plateau_C4",
            "minimizer_diffusion",
            "minimizer_Peq",
            "minimizer_long_time_PDF",
            "minimizer_short_time_PDF",
            "minimizer_F_eq",
        ]

        self.not_need_getter = [
            "minimizer_long_time_PDF",
            "minimizer_short_time_PDF",
        ]

        self._method = method
        self.set_method()

        self.to_vary = to_vary
        self._Model = Model
        self._analysis = analysis
        self.dt = self.analysis.Data.dt
        self._range_MSD_short = range_MSD_short
        self._range_plateau_MSD = range_plateau_MSD
        self._range_C4_short = range_C4_short
        if range_diffusion == None:
            self._range_diffusion = analysis.range_D
        else:
            self._range_diffusion = range_diffusion

        if range_Peq == None:
            self._range_Peq = analysis.range_pdf
        else:
            self._range_Peq = range_diffusion

        if range_Feq == None:
            self._range_Feq = analysis.range_F_eq
        else:
            self._range_F_eq = range_Feq

        if to_compute == "all":
            self.to_compute = self.available_minimizer
        else:
            for i in to_compute:
                if i not in self.available_minimizer:
                    raise ValueError(
                        "Please select minimizers in " + str(self.available_minimizer)
                    )
            self.to_compute = to_compute

        for i in self.to_compute:
            if i in self.not_need_getter:
                continue
            else:
                fun = "_get_" + i[10:]
                getattr(self, fun)()

    def set_method(self):
        if self.method not in self.all_method:
            raise ValueError("Please select a method in " + str(all_method) + ".")
        match self.method:
            case "sum":
                self.func = np.nansum
            case "mean":
                self.func = np.nanmean

    ### Gather experimental points

    def _get_MSD_short_time(self):
        # retrive the points over witch we want to fit the MSD

        I1 = int(
            np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[0])[0]
        )
        I2 = int(
            np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[1])[0]
        )
        axis = self.analysis.axis

        # retrieve the experimental points
        self.t_MSD_short = self.analysis.MSD_t[I1:I2] * self.dt

        self.MSD_short_exp = np.zeros((len(self.t_MSD_short), len(axis)))
        self.MSD_short_exp[:, 0] = self.analysis.MSD["x"][I1:I2]
        self.MSD_short_exp[:, 1] = self.analysis.MSD["y"][I1:I2]
        self.MSD_short_exp[:, 2] = self.analysis.MSD["z"][I1:I2]

    def _get_C4_short_time(self):
        # retrive the points over witch we want to fit the MSD

        I1 = int(np.argwhere(self.analysis.MSD_t * self.dt > self.range_C4_short[0])[0])
        I2 = int(np.argwhere(self.analysis.MSD_t * self.dt > self.range_C4_short[1])[0])

        axis = self.analysis.axis

        # retrieve the experimental points
        self.t_C4_short = self.analysis.C4_t[I1:I2] * self.dt

        self.C4_short_exp = np.zeros((len(self.t_C4_short), len(axis)))
        self.C4_short_exp[:, 0] = self.analysis.C4["x"][I1:I2]
        self.C4_short_exp[:, 1] = self.analysis.C4["y"][I1:I2]
        self.C4_short_exp[:, 2] = self.analysis.C4["z"][I1:I2]

    def _get_plateau_MSD(self):

        I1 = int(
            np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[0])[0]
        )
        I2 = int(
            np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[1])[0]
        )

        self.plateau_exp = self.analysis.MSD["z"][I1:I2]

    def _get_plateau_C4(self):

        """ """

        I1 = int(np.argwhere(self.analysis.C4_t * self.dt > self.range_C4_short[0])[0])
        I2 = int(np.argwhere(self.analysis.C4_t * self.dt > self.range_C4_short[1])[0])

        self.plateau_C4_exp = self.analysis.MSD["z"][I1:I2]

    def _get_diffusion(self):

        """
        Retrieve the experimetal diffusion data points
        """

        if self.range_diffusion == self.analysis.range_D:
            Dx, Dy, Dz, z_D = (
                self.analysis.Dx,
                self.analysis.Dy,
                self.analysis.Dz,
                self.analysis.z_D,
            )
        else:
            I1 = int(np.argwhere(self.analysis.z_D * self.dt > self.range_diffusion[0])[0])
            I2 = int(np.argwhere(self.analysis.z_D * self.dt > self.range_diffusion[1])[0])

            Dx, Dy, Dz, z_D = (
                self.analysis.Dx[I1:I2],
                self.analysis.Dy[I1:I2],
                self.analysis.Dz[I1:I2],
                self.analysis.z_D[I1:I2],
            )

        self.z_Diffusion = z_D
        self.Diffusion = np.zeros((len(z_D), 3))
        self.Diffusion[:, 0] = Dx
        self.Diffusion[:, 1] = Dy
        self.Diffusion[:, 2] = Dz

    def _get_Peq(self):
        """
        Retrieve the experimental Peq Data points
        """

        if self.range_Peq == self.analysis.range_pdf or self.range_Peq == None:
            z_Peq, Peq = self.analysis.z_pdf_z, self.analysis.pdf_z
        else:
            I1 = int(np.argwhere(self.analysis.z_pdf_z > self.range_Peq[0])[0])
            I2 = int(np.argwhere(self.analysis.z_pdf_z > self.range_Peq[1])[0])

            z_Peq, Peq = self.analysis.z_pdf_z[I1:I2], self.analysis.pdf_z[I1:I2]

        self.z_Peq = z_Peq
        self.P_eq = Peq

    def _get_F_eq(self):
        """
        Retrieve the experimental Peq Data points
        """

        if self.range_Feq == self.analysis.range_F_eq or self.range_Feq == None:
            z_Feq, Feq = self.analysis.z_F_eq, self.analysis.F_eq
        else:
            I1 = int(np.argwhere(self.analysis.z_F_eq > self.range_Feq[0])[0])
            I2 = int(np.argwhere(self.analysis.z_F_eq > self.range_Feq[1])[0])

            z_Feq, Feq = self.analysis.z_F_eq[I1:I2], self.analysis.F_eq[I1:I2]

        self.z_Feq = z_Feq
        self.F_eq = Feq

    def minimizer_MSD_short_time(self):
        """
        Compute the squared relative error between experimental and theoritical
        average diffusion coefficient.
        """

        # Compute the theory (same over x and y)
        MSD_th = np.zeros((len(self.t_MSD_short), len(self.analysis.axis)))
        t_exp = self.t_MSD_short
        MSD_th[:, 0] = self.Model.MSD_short_time(t_exp, axis="x")
        MSD_th[:, 1] = MSD_th[:, 0]
        MSD_th[:, 2] = self.Model.MSD_short_time(t_exp, axis="z")

        # Compute the squared relative error and return it

        return self.func((self.MSD_short_exp - MSD_th) ** 2 / MSD_th**2)

    def minimizer_C4_short_time(self):
        """
        Compute the squared relative error between experimental and theoritical
        average diffusion coefficient.
        """

        # Compute the theory (same over x and y)
        C4_th = np.zeros((len(self.t_C4_short), len(self.analysis.axis)))
        t_exp = self.t_C4_short
        C4_th[:, 0] = self.Model.C4_short_time(t_exp, axis="x")
        C4_th[:, 1] = C4_th[:, 0]
        C4_th[:, 2] = self.Model.C4_short_time(t_exp, axis="z")

        # Compute the squared relative error and return it

        return self.func((self.C4_short_exp - C4_th) ** 2 / C4_th**2)

    def minimizer_plateau_MSD(self):
        """
        Conmpute the squared relative error between experimental and theoritical
        long time plateau on the z MSD at long time.
        """
        plateau_th = np.array([self.Model.plateau_MSD()] * len(self.plateau_exp))

        return self.func((self.plateau_exp - plateau_th) ** 2 / plateau_th)

    def minimizer_plateau_C4(self):
        """
        Compute the squared relative error between experimental and theoritical
        long time plateau on the z MSD at long time.
        """
        plateau_th = np.array([self.Model.plateau_C4()] * len(self.plateau_C4_exp))

        return self.func((self.plateau_C4_exp - plateau_th) ** 2 / plateau_th)

    def minimizer_diffusion(self):
        """
        Compute the squared relative error between experimental and theoritical
        diffusion profile.
        """

        # Compute the theoritical diffusion profile

        z_exp = self.z_Diffusion
        Diffusion_th = np.zeros((len(z_exp), 3))

        Diffusion_th[:, 0] = self.Model.Dx_off(z_exp)
        Diffusion_th[:, 1] = Diffusion_th[:, 0]
        Diffusion_th[:, 2] = self.Model.Dz_off(z_exp)

        # Compute the squared relative error

        return self.func((self.Diffusion - Diffusion_th) ** 2 / Diffusion_th**2)

    def minimizer_Peq(self):
        """
        Compute the squared relative error between experimental and theoritical
        equilibrium height distribution .
        """
        z_exp = self.z_Peq

        # Compute the theoritical z_Peq

        P_eq_th = self.Model.P_0_off(z_exp)

        # Compute the squared relative errorbar

        return self.func((self.P_eq - P_eq_th) ** 2 / P_eq_th**2)

    def minimizer_long_time_PDF(self):
        """
        Compute the squared relative error between experimental and theoritical
        long time perpendicular displacement distribution.
        """

        pdf_long_time_th = self.Model.long_time_pdf(self.analysis.bins_centers_long_t)
        return np.sum(
            (self.analysis.pdf_long_t - pdf_long_time_th) ** 2 / pdf_long_time_th**2
        )

    def minimizer_short_time_PDF(self):
        """
        Compute the squared relative error between experimental and theoritical
        short time perpendicular displacement distribution.
        """

        # Retrieve computation time
        keys = self.analysis.short_time_PDF_Dx.keys()
        err = []

        # Computing each relative error for each time
        for i in keys:
            dt = float(i) * self.dt
            dict_pdf_x = self.analysis.short_time_PDF_Dx[i]
            dict_pdf_y = self.analysis.short_time_PDF_Dy[i]
            dict_pdf_z = self.analysis.short_time_PDF_Dz[i]

            pdf_x_th = self.Model.P_D_short_time(dict_pdf_x["bin_center"], dt, axis="x")
            pdf_y_th = self.Model.P_D_short_time(dict_pdf_y["bin_center"], dt, axis="y")
            pdf_z_th = self.Model.P_D_short_time(dict_pdf_z["bin_center"], dt, axis="z")

            err.append(self.func((dict_pdf_x["PDF"] - pdf_x_th) ** 2 / pdf_x_th**2))
            err.append(self.func((dict_pdf_x["PDF"] - pdf_z_th) ** 2 / pdf_z_th**2))
            err.append(self.func((dict_pdf_x["PDF"] - pdf_y_th) ** 2 / pdf_y_th**2))

        err = np.array(err)
        err[err > 2] = 0  ## remove experiment that did not work at all
        return self.func(err)

    def minimizer_F_eq(self):
        """
        Compute the squared relative error between experimental and theoritical
        conservative force.
        """
        z_exp = self.z_Feq

        # Compute the theoritical Feq

        F_eq_th = self.Model.Conservative_Force(z_exp)

        # Compute the squared relative errorbar

        return self.func((self.F_eq - F_eq_th) ** 2 / F_eq_th**2)

    ### Computing minimizer

    def Compute(self, verbose=False):
        """
        Compute the total minimizer.
        The computed minimizers are stored in self.to_compute.
        The different output are just
        """
        minimizer = 0

        for i in self.to_compute:

            err = getattr(self, i)()
            if verbose:
                print(i + " = " + str(err))
            minimizer += err

        if verbose:
            print("Total minimizer = " + str(minimizer))

        return minimizer

    def _minimize(self, x):
        """
        Actual function to minimize
        """
        for n, i in enumerate(self.to_vary):
            setattr(self.Model, i, x[n])

        return self.Compute()

    def Optimizer(self, kwargs={}):
        """
        Find the minimium of the total minimizer by varing the variable listed in self.to_vary
        Takes the same arguments as scipy.optimize.minimize function.
        """

        x0 = [getattr(self.Model, i) for i in self.to_vary]
        self.res = minimize(self._minimize, x0, **kwargs)

        print("done")

    #########   Ploting tools ################
    def plot(self, info: str, ax=None):
        """Simplifying all the plots."""
        info = info.lower()
        needs_plot = False

        if ax is None:
            plt.ioff()
            fig = plt.figure()
            plt.ion()
            ax = plt.gca()
            needs_plot = True

        match info:
            case "msd":
                t_exp = self.analysis.MSD_t * self.analysis.Data.dt

                for i in self.analysis.axis:

                    ax.loglog(t_exp, self.analysis.MSD[i], "o", label="$" + i + "$")
                    ax.plot(t_exp, self.Model.MSD_short_time(t_exp, axis=i), color="k")

                ax.plot(
                    t_exp[-30:],
                    [self.Model.plateau_MSD()] * 30,
                    color="k",
                    linewidth=5,
                    alpha=0.5,
                )

                ax.legend(frameon=False)
                ax.set(xlabel="$t$ (s)", ylabel="MSD (m$^2$)")

            case "pdf":

                ax.semilogy(self.analysis.z_pdf_z, self.analysis.pdf_z, "o")
                ax.plot(self.z_Peq, self.Model.P_0_off(self.z_Peq), color="k")
                ax.set(xlabel="$z$ ($\mu$m)", ylabel="PDF (m$^{-1}$)")

            case "f_eq":
                ax.semilogx(self.analysis.z_F_eq, self.analysis.F_eq * 1e15, "o")
                ax.plot(
                    self.z_Feq, self.Model.Conservative_Force(self.z_Feq) * 1e15, color="k"
                )
                ax.set(xlabel="$z$ (m)", ylabel="$F$ (fN)")

            case "c4":

                t_exp = self.analysis.C4_t * self.analysis.Data.dt
                for i in self.analysis.axis:
                    ax.loglog(t_exp, self.analysis.C4[i], "o", label="$" + i + "$")
                    ax.plot(t_exp, self.Model.C4_short_time(t_exp, axis=i), color="k")

                ax.plot(
                    t_exp[-30:],
                    [self.Model.plateau_C4()] * 30,
                    color="k",
                    linewidth=5,
                    alpha=0.5,
                )

                ax.legend(frameon=False)
                ax.set(xlabel="$t$ (s)", ylabel="C4 (m$^4$)")

            case "short_time_pdf_x":

                I1 = int(self.analysis.t_sPDF[0] * self.analysis.Data.fps)
                if I1 == 0:
                    I1 += 1
                I2 = int(self.analysis.t_sPDF[1] * self.analysis.Data.fps)
                I = np.arange(I1, I2)

                for i in I:
                    PDF = self.analysis.short_time_PDF_Dx[str(i)]
                    dt = float(i) * self.dt
                    ax.semilogy(
                        PDF["bin_center"],
                        PDF["PDF"],
                        "o",
                        label="$\Delta t  = " + str(i / self.analysis.Data.fps) + "$ s",
                    )

                    ax.plot(PDF["bin_center"], self.Model.P_D_short_time(PDF["bin_center"], dt, axis="x"), color = "k")
                ax.legend(frameon=False)
                ax.set(xlabel="$\Delta x$ (m)", ylabel="PDF (m$^-1$)")
            case "short_time_pdf_z":

                I1 = int(self.analysis.t_sPDF[0] * self.analysis.Data.fps)
                if I1 == 0:
                    I1 += 1
                I2 = int(self.analysis.t_sPDF[1] * self.analysis.Data.fps)
                I = np.arange(I1, I2)

                for i in I:
                    PDF = self.analysis.short_time_PDF_Dz[str(i)]
                    dt = float(i) * self.dt
                    ax.semilogy(
                        PDF["bin_center"],
                        PDF["PDF"],
                        "o",
                        label="$\Delta t  = " + str(i / self.analysis.Data.fps) + "$ s",
                    )
                    ax.plot(PDF["bin_center"], self.Model.P_D_short_time(PDF["bin_center"], dt, axis="z"), color = "k")
                ax.legend(frameon=False)
                ax.set(xlabel="$\Delta z$ (m)", ylabel="PDF (m$^-1$)")
            case "long_time_pdf":

                ax.errorbar(
                    self.analysis.bins_centers_long_t,
                    self.analysis.pdf_long_t,
                    yerr=self.analysis.err_long_t,
                    fmt="o",
                    barsabove=False,
                    linestyle="",
                    ecolor="k",
                    capsize=4,
                )
                ax.plot(
                    self.analysis.bins_centers_long_t,
                    self.Model.long_time_pdf(self.analysis.bins_centers_long_t),
                    color="k",
                    zorder=10,
                )

                ax.semilogy()
                ax.set(xlabel="$\Delta z$ (m)", ylabel="PDF (m$^-1$)")

            case "local_d":
                z_D = self.analysis.z_D
                Dx = self.analysis.Dx
                Dy = self.analysis.Dy
                Dz = self.analysis.Dz
                Do = self.analysis.Do
                z_exp = self.z_Diffusion
                ax.loglog(z_D, Dz / Do, "o", label=r"$D_\bot$")
                ax.plot(z_D, (Dx + Dy) / 2 / Do, "o", label=r"D_\\parallel")
                ax.plot(z_exp, self.Model.Dx_off(z_exp)/Do, color = "k")
                ax.plot(z_exp, self.Model.Dz_off(z_exp)/Do, color = "k")
                ax.legend(frameon=False)

                ax.set(xlabel="$z$ (m)", ylabel="$D/ D_0$ (m$^2$s$^-1$)")
        if needs_plot:
            plt.show()

    def plot_result(self):
        plt.ioff()
        plt.figure(figsize=(10, 10))
        plt.ion()

        ### MSD

        ax1 = plt.subplot(421)
        self.plot("msd", ax1)

        ###

        ax2 = plt.subplot(422)
        self.plot("C4", ax2)

        ax3 = plt.subplot(423)
        self.plot("short_time_pdf_x", ax3)
        ax3.semilogy()

        ax4 = plt.subplot(424)
        self.plot("short_time_pdf_z", ax4)

        ax5 = plt.subplot(425)
        self.plot("local_D", ax5)

        ax6 = plt.subplot(426)
        self.plot("long_time_pdf", ax6)

        ax7 = plt.subplot(427)
        self.plot("pdf", ax7)

        ax8 = plt.subplot(428)
        self.plot("F_eq", ax8)

        plt.tight_layout()
        plt.show()

    ### Getter and setters

    @property
    def range_Feq(self):
        return self._range_Feq

    @range_Feq.setter
    def range_Feq(self, range_Feq):
        self._range_Feq = range_Feq
        self._get_Feq()

    ###

    @property
    def Model(self):
        return self._Model

    @Model.setter
    def Model(self, Model):
        self._Model = Model

    ###

    @property
    def analysis(self):
        return self._analysis

    @analysis.setter
    def analysis(self, analysis):
        self._analysis = analysis

    ###

    @property
    def range_MSD_short(self):
        return self._range_MSD_short

    @range_MSD_short.setter
    def range_MSD_short(self, range_MSD_short):
        self._range_MSD_short = range_MSD_short
        self._get_MSD_short()

    ###

    @property
    def range_plateau_MSD(self):
        return self._range_plateau_MSD

    @range_plateau_MSD.setter
    def range_plateau_MSD(self, range_plateau_MSD):
        self._range_plateau_MSD = range_plateau_MSD
        self._get_MSD_plateau()

    ###

    @property
    def range_C4_short(self):
        return self._range_C4_short

    @range_C4_short.setter
    def range_C4_short(self, range_C4_short):
        self._range_C4_short = range_C4_short
        self._get_C4_short()

    ###

    @property
    def range_plateau_MSD(self):
        return self._range_plateau_MSD

    @range_plateau_MSD.setter
    def range_plateau_MSD(self, range_plateau_MSD):
        self._range_plateau_MSD = range_plateau_MSD
        self._get_C4_plateau()

    ###

    @property
    def range_diffusion(self):
        return self._range_diffusion

    @range_diffusion.setter
    def range_diffusion(self, range_diffusion):
        self._range_diffusion = range_diffusion
        self._get_diffusion()

    ###

    @property
    def range_Peq(self):
        return self._range_Peq

    @range_Peq.setter
    def range_Peq(self, range_Peq):
        self._range_Peq = range_Peq
        self._get_Peq()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        self._method = method
        self.set_method()
