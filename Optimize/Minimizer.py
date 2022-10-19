class Minimizer:
    def __init__(
        self,
        Model,
        analysis,
        range_MSD_short=(1e-2, 1),
        range_plateau_MSD=(1e-2, 1e-1),
        range_C4_short=(1e-2, 1e-1),
        range_diffusion=None,
        range_Peq=None,
        range_Feq=None,
    ):

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
            self._range_Peq = analysis.range_D
        else:
            self._range_Peq = range_diffusion

        if range_Feq == None:
            self._range_Feq = analysis.range_F_eq
        else:
            self._range_F_eq = range_Feq

        self.available_minimizer = [
            "minimizer_MSD_short_time",
            "minimizer_plateau_MSD",
            "minimizer_plateau_C4",
            "minimizer_diffusion",
            "minimizer_Peq",
            "minimizer_long_time_PDF",
            "minimizer_short_time_PDF",
            "minimizer_F_eq",
        ]

    ### Getter and setters

    @parameter
    def range_Feq(self):
        return self._range_Feq

    @range_Feq.setter
    def range_Feq(self, range_Feq):
        self._range_Feq = range_Feq
        self._get_Feq()

    ###

    @parameter
    def Model(self):
        return self._Model

    @Model.setter
    def Model(self, Model):
        self._Model = Model

    ###

    @parameter
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

    @parameter
    def range_plateau_MSD(self):
        return self._range_plateau_MSD

    @range_plateau_MSD.setter
    def range_plateau_MSD(self, range_plateau_MSD):
        self._range_plateau_MSD = range_plateau_MSD
        self._get_MSD_plateau()

    ###

    @parameter
    def range_C4_short(self):
        return self._range_C4_short

    @range_C4_short.setter
    def range_C4_short(self, range_C4_short):
        self._range_C4_short = range_C4_short
        self._get_C4_short()

    ###

    @parameter
    def range_plateau_MSD(self):
        return self._range_plateau_MSD

    @range_plateau_MSD.setter
    def range_plateau_MSD(self, range_plateau_MSD):
        self._range_plateau_MSD = range_plateau_MSD
        self._get_C4_plateau()

    ###

    @parameter
    def range_diffusion(self):
        return self._range_diffusion

    @range_diffusion.setter
    def range_diffusion(self, range_diffusion):
        self._range_diffusion = range_diffusion
        self._get_diffusion()

    ###

    @parameter
    def range_Peq(self):
        return self._range_Peq

    @range_Peq.setter
    def range_Peq(self, range_Peq):
        self._range_Peq = range_Peq
        self._get_Peq()

    ### Gather experimental points

    def _get_MSD_short(self):
        # retrive the points over witch we want to fit the MSD

        I1 = np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[0])[0]
        I2 = np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[1])[0]

        axis = self.analysis.axis

        # retrieve the experimental points
        self.t_MSD_short = self.analysis.MSD_t[I1:I2] * self.dt

        self.MSD_short_exp = np.zeros((len(x_exp)), len(axis))
        self.MSD_short_exp[:, 0] = self.analysis.MSD["x"][I1:I2]
        self.MSD_short_exp[:, 1] = self.analysis.MSD["y"][I1:I2]
        self.MSD_short_exp[:, 2] = self.analysis.MSD["z"][I1:I2]

    def _get_C4_short(self):
        # retrive the points over witch we want to fit the MSD

        I1 = np.argwhere(self.analysis.MSD_t * self.dt > self.range_C4_short[0])[0]
        I2 = np.argwhere(self.analysis.MSD_t * self.dt > self.range_C4_short[1])[0]

        axis = self.analysis.axis

        # retrieve the experimental points
        self.t_C4_short = self.analysis.C4_t[I1:I2] * self.dt

        self.C4_short_exp = np.zeros((len(x_exp)), len(axis))
        self.C4_short_exp[:, 0] = self.analysis.C4["x"][I1:I2]
        self.C4_short_exp[:, 1] = self.analysis.C4["y"][I1:I2]
        self.C4_short_exp[:, 2] = self.analysis.C4["z"][I1:I2]

    def _get_MSD_plateau(self):

        I1 = np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[0])[0]
        I2 = np.argwhere(self.analysis.MSD_t * self.dt > self.range_MSD_short[1])[0]

        self.plateau_exp = self.analysis.MSD["z"][I1:I2]

    def _get_C4_plateau(self):

        """ """

        self.MSD_short_exp = np.zeros((len(x_exp)), 3)
        self.plateau_C4 = self.analysis.C4["z"][I1:I2]

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
            I1 = np.argwhere(self.analysis.z_D * self.dt > self.range_D[0])[0]
            I2 = np.argwhere(self.analysis.z_D * self.dt > self.range_D[1])[0]

            Dx, Dy, Dz, z_D = (
                self.analysis.Dx[I1:I2],
                self.analysis.Dy[I1:I2],
                self.analysis.Dz[I1:I2],
                self.analysis.z_D[I1:I2],
            )

        self.z_Diffusion = z_D
        self.Diffusion = np.zeros((len(z_D)), 3)
        self.Diffusion[:, 0] = Dx
        self.Diffusion[:, 1] = Dy
        self.Diffusion[:, 2] = Dz

    def _get_Peq(self):
        """
        Retrieve the experimental Peq Data points
        """

        if self.range_Peq == self.analysis.range_pdf:
            z_Peq, Peq = self.analysis.z_pdf_z, self.analysis.pdf_z
        else:
            I1 = np.argwhere(self.analysis.z_pdf_z > self.range_pdf[0])[0]
            I2 = np.argwhere(self.analysis.z_pdf_z > self.range_pdf[1])[0]

            z_Peq, Peq = self.analysis.z_pdf_z[I1:I2], self.analysis.pdf_z[I1:I2]

        self.z_Peq = z_Peq
        self.P_eq = Peq

    def _get_Feq():
        """
        Retrieve the experimental Peq Data points
        """

        if self.range_Feq == self.analysis.range_Feq:
            z_Feq, Feq = self.analysis.z_F_eq, self.analysis.F_eq
        else:
            I1 = np.argwhere(self.analysis.z_F_eq > self.range_Feq[0])[0]
            I2 = np.argwhere(self.analysis.z_F_eq > self.range_Feq[1])[0]

            z_Peq, Peq = self.analysis.z_F_eq[I1:I2], self.analysis.F_eq[I1:I2]

        self.z_Feq = z_Peq
        self.F_eq = Peq

    def minimizer_MSD_short_time(self):
        """
        Compute the squared relative error between experimental and theoritical
        average diffusion coefficient.
        """

        # Compute the theory (same over x and y)
        MSD_th = np.zeros((len(x_exp)), len(axis))
        t_exp = self.t_MSD_short
        MSD_th[:, 0] = self.Model.MSD_short_time(self, t_exp, axis="x")
        MSD_th[:, 1] = MSD_th[:, 0]
        MSD_th[:, 2] = self.Model.MSD_short_time(self, t_exp, axis="z")

        # Compute the squared relative error and return it

        return np.sum((self.MSD_short_exp - MSD_th) ** 2 / MSD_th**2)

    def minimizer_plateau_MSD(self):
        """
        Conmpute the squared relative error between experimental and theoritical
        long time plateau on the z MSD at long time.
        """
        plateau_th = np.array([self.analysis.plateau_MSD()] * len(self.plateau_exp))

        return np.sum((self.plateau_exp - plateau_th) ** 2 / plateau_th)

    def minimizer_plateau_C4(self):
        """
        Compute the squared relative error between experimental and theoritical
        long time plateau on the z MSD at long time.
        """
        plateau_th = np.array([self.analysis.plateau_C4()] * len(self.plateau_exp))

        return np.sum((self.plateau_exp - plateau_th) ** 2 / plateau_th)

    def minimizer_diffusion(self):
        """
        Compute the squared relative error between experimental and theoritical
        diffusion profile.
        """

        # Compute the theoritical diffusion profile

        z_exp = self.z_Diffusion
        Diffusion_th = np.zeros((len(z_exp)), 3)

        Diffusion_th[:, 0] = self.Model.Dx_off(self, z_exp)
        Diffusion_th[:, 1] = MSD_th[:, 0]
        Diffusion_th[:, 2] = self.Model.Dz_off(self, z_exp)

        # Compute the squared relative error

        return np.sum((self.Diffusion - Diffusion_th) ** 2 / Diffusion_th**2)

    def minimizer_Peq(self):
        """
        Compute the squared relative error between experimental and theoritical
        equilibrium height distribution .
        """
        z_exp = self.z_Peq

        # Compute the theoritical z_Peq

        P_eq_th = self.Model.P_0_off(z_exp)

        # Compute the squared relative errorbar

        return np.sum((self.P_eq - P_eq_th) ** 2 / P_eq_th**2)

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
        err = 0

        # Computing each relative error for each time
        for i in keys:
            dt = float(i) * self.dt
            dict_pdf_x = self.analysis.short_time_Dx[i]
            dict_pdf_y = self.analysis.short_time_Dy[i]
            dict_pdf_z = self.analysis.short_time_Dz[i]

            pdf_x_th = self.Model.P_D_short_time(dict_pdf_x["bin_center"], dt, axis="x")
            pdf_y_th = self.Model.P_D_short_time(dict_pdf_y["bin_center"], dt, axis="x")
            pdf_z_th = self.Model.P_D_short_time(dict_pdf_z["bin_center"], dt, axis="z")

            err += (dict_pdf_x["PDF"] - pdf_x_th) ** 2 / pdf_x_th
            err += (dict_pdf_x["PDF"] - pdf_z_th) ** 2 / pdf_z_th
            err += (dict_pdf_x["PDF"] - pdf_y_th) ** 2 / pdf_y_th

        return err

    def minimizer_F_eq(self):
        """
        Compute the squared relative error between experimental and theoritical
        conservative force.
        """
        z_exp = self.z_Peq

        # Compute the theoritical Feq

        F_eq_th = self.Model.Conservative_Force(z_exp)

        # Compute the squared relative errorbar

        return np.sum((self.F_eq - F_eq_th) ** 2 / F_eq_th**2)


class Optimizer(Minimizer):
    def __init__(
        self,
        Model,
        analysis,
        self.to_vary = None
        range_MSD_short=(1e-2, 1),
        range_plateau_MSD=(1e-2, 1e-1),
        range_C4_short=(1e-2, 1e-1),
        range_diffusion=None,
        range_Peq=None,
        range_Feq=None,
    ):

        self.super().__init__(
            Model,
            analysis,
            range_MSD_short=(1e-2, 1),
            range_plateau_MSD=(1e-2, 1e-1),
            range_C4_short=(1e-2, 1e-1),
            range_diffusion=None,
            range_Peq=None,
            range_Feq=None,
        )

        if to_vary == None:
            self._to_vary = self.available_minimizer



    @parameter
    def to_vary(self):
        return self._to_vary

    @to_vary.setter
    def to_vary(self, to_vary):
        self._to_vary = to_vary
