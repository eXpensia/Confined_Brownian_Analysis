import numpy as np
from StochasticForceInference.fun_SFI import Compute_diffusion
from tqdm import tqdm
import matplotlib.pyplot as plt


class Data_analysis:
    """Class permitting the automatic analysis of Brownian trajectories
    the goal being to extract the non-conservative forces."""

    def __init__(
        self,
        Data,
        verbose=True,
        MSD_bins: int = 100,  # Number of PDF bins
        t_LMSD: (float, float) = (20.0, 25),  # Time used to compute the plateau
        t_sPDF: (float, float) = (0.01, 0.05),  # Short time PDF dispalcement
        range_pdf: (float, float) = (1e-8, 2.5e-6),  # Range of the height PDF
        num_pdf: int = 70,  # number of bins in the height PDF
        range_F_eq: (float, float) = (1e-8, 2e-6),
        num_F_eq: int = 50,
        SPDF_bins: int = 40,  # Short time PDF bins
        t_Lpdf: (float, float) = (18.0, 20.0),  # time for the long time PDF
        range_D: (float, float) = (1e-9, 2e-6),  # range of computation of Ronceray
        N_local_D: int = 200,  # number of points for the diffusion computation
        ordre_D: int = 3,  # Ronceray polynomial order
        LPDF_bins: int = 50,  # number of bins in the long time PDF
        Do=4e-21 / (6 * np.pi * 0.001 * 1.50e-6),
    ):
        """Loading the data une the Data class."""
        self.Data = Data
        self._Data = Data
        self.axis = ["x", "y", "z"]
        if np.max(self.Data[:]) > 1:
            self.Data *= 1e-6  # Putting everything in microns.

        self._MSD_bins = MSD_bins
        self._t_LMSD = t_LMSD
        self._range_pdf = range_pdf
        self._num_pdf = num_pdf
        self._t_sPDF = t_sPDF
        self._SPDF_bins = SPDF_bins
        self._t_Lpdf = t_Lpdf
        self._LPDF_bins = LPDF_bins
        self._N_local_D = N_local_D
        self._range_D = range_D
        self._ordre_D = ordre_D

        self._range_F_eq = range_F_eq
        self._num_F_eq = num_F_eq

        self.verbose = verbose
        self.Do = Do

    ### Getters and setters ###
    # The point of the getters and setters is to compute automatically
    # the observables when changing the computation variables.
    ###

    @property
    def MSD_bins(self):
        return self._MSD_bins

    @MSD_bins.setter
    def MSD_bins(self, MSD_bins):
        self._MSD_bins = MSD_bins
        self._MSD()
        self._C4()

    ###

    @property
    def t_LMSD(self):
        return self._t_LMSD

    @t_LMSD.setter
    def t_LMSD(self, MSD_bins):
        self._t_LMSD = t_LMSD
        self._MSD()

    ###

    @property
    def range_pdf(self):
        return self._range_pdf

    @range_pdf.setter
    def range_pdf(self, range_pdf):
        self._range_pdf = range_pdf
        self.PDF_z()

    ###

    @property
    def num_pdf(self):
        return self._num_pdf

    @num_pdf.setter
    def num_pdf(self, num_pdf):
        self._num_pdf = num_pdf
        self.PDF_z()

    ###

    @property
    def range_F_eq(self):
        return self._range_F_eq

    @range_F_eq.setter
    def range_F_eq(self, range_F_eq):
        self._range_F_eq = range_F_eq
        self.C_F_eq()

    ###

    @property
    def num_F_eq(self):
        return self._num_F_eq

    @num_F_eq.setter
    def num_F_eq(self, num_F_eq):
        self._num_F_eq = num_F_eq
        self.C_F_eq()

    ###

    @property
    def t_sPDF(self):
        return self._t_sPDF

    @t_sPDF.setter
    def t_sPDF(self, t_sPDF):
        self._t_sPDF = t_sPDF
        self.Short_time_PDF()

    ###

    @property
    def SPDF_bins(self):
        return self._SPDF_bins

    @SPDF_bins.setter
    def SPDF_bins(self, SPDF_bins):
        self._SPDF_bins = SPDF_bins
        self.Short_time_PDF()

    ###

    @property
    def t_Lpdf(self):
        return self._t_Lpdf

    @t_Lpdf.setter
    def t_Lpdf(self, t_Lpdf):
        self._t_Lpdf = t_Lpdf
        self.Large_Dt_PDF()

    ###

    @property
    def LPDF_bins(self):
        return self._LPDF_bins

    @LPDF_bins.setter
    def LPDF_bins(self, LPDF_bins):
        self._LPDF_bins = LPDF_bins
        self.Large_Dt_PDF()

    ###

    @property
    def N_local_D(self):
        return self._N_local_D

    @N_local_D.setter
    def N_local_D(self, N_local_D):
        self._N_local_D = N_local_D
        self.local_diffusion()

    ###

    @property
    def range_D(self):
        return self._range_D

    @range_D.setter
    def range_D(self, range_D):
        self._range_D = range_D
        self.local_diffusion()

    ###

    @property
    def ordre_D(self):
        return self._ordre_D

    @ordre_D.setter
    def ordre_D(self, ordre_D):
        self._ordre_D = ordre_D
        self.local_diffusion()

    ######## Methods

    def refresh_Data(self):
        "Helper method  to reset the change to Data"
        self.Data = self._Data

    def computing_all(self):
        if self.verbose:
            print("Computing MSD")

        self._MSD()

        if self.verbose:
            print("Computing long time MSD")

        self.Long_time_MSD()

        if self.verbose:
            print("Computing PDF")

        self.PDF_z()

        if self.verbose:
            print("Computing the conservative forces")

        self.C_F_eq()

        if self.verbose:
            print("Computing 4th cumulant")

        self.C4 = self._C4()

        if self.verbose:
            print("Computing PDF at small times")

        self.Short_time_PDF()

        if self.verbose:
            print("Computing PDF at large time")

        self.Large_Dt_PDF()

        if self.verbose:
            print("Computing the local diffusion")

        self.local_diffusion()

    def _pdf(
        self, data: np.ndarray, bins: int = 10, density: bool = True
    ) -> (np.ndarray, np.ndarray):
        """Function to compute a pdf using the histogram function and retrieving the position of the middle of the bins."""
        pdf, bins_edge = np.histogram(data, bins=bins, density=density)
        bins_center = (bins_edge[:-1] + bins_edge[1:]) / 2

        return pdf, bins_center

    def logarithmic_hist(
        self, data: np.ndarray, begin: int, stop: int, num: int = 50, base: int = 10
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """Function to compute a pdf using  logspaced bins"""
        if begin == 0:
            beg = stop / num
            bins = np.logspace(
                np.log(beg) / np.log(base),
                np.log(stop) / np.log(base),
                num - 1,
                base=base,
            )
            widths = bins[1:] - bins[:-1]
            bins = np.cumsum(widths[::-1])
            bins = np.concatenate(([0], bins))
            widths = bins[1:] - bins[:-1]

        else:
            bins = np.logspace(
                np.log(begin) / np.log(base),
                np.log(stop) / np.log(base),
                num,
                base=base,
            )
            widths = bins[1:] - bins[:-1]

        hist, bins = np.histogram(data, bins=bins, density=True)
        # normalize by bin width
        bins_center = (bins[1:] + bins[:-1]) / 2

        return bins_center, widths, hist

    def PDF_z(self, base: int = 10, output=False):
        """Subfunction to compute the pdf of height"""
        self.z_pdf_z, _, self.pdf_z = self.logarithmic_hist(
            self.Data.z, *self.range_pdf, self.num_pdf, base
        )
        if output:
            return self.z_pdf_z, self.pdf_z

    def C_F_eq(self):
        """Computing the conservative forces by computing the derivative of the log of the height position."""
        z_pdf, _, pdf_z = self.logarithmic_hist(
            self.Data.z, *self.range_F_eq, self.num_F_eq, base=10
        )

        self.F_eq = np.gradient(np.log(pdf_z), z_pdf) * 4e-21
        self.z_F_eq = z_pdf

    def _MSD_1D(
        self,
        t: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """Actual computation of the MSD"""

        MSD = np.zeros(len(t))
        for n, i in enumerate(tqdm(t)):
            MSD[n] = np.nanmean((x[: -int(i)] - x[int(i) :]) ** 2)

        return MSD

    def _MSD(self) -> dict:
        """Actual computation of the MSD"""

        self.MSD = {}
        self.MSD_t = np.logspace(
            np.log(1) / np.log(10),
            np.log(len(self.Data) / 10) / np.log(10),
            num=self.MSD_bins,
        )
        self.MSD_t = np.unique(self.MSD_t.astype(int))
        for n, i in enumerate(self.axis):
            if self.verbose:
                print("Computing MSD on " + i)
            self.MSD[i] = self._MSD_1D(self.MSD_t, self.Data[:, n])

        return self.MSD

    def _C4_1D(
        self,
        t: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """Actual computation of the C4"""

        C4 = np.zeros(len(t))
        for n, i in enumerate(tqdm(t)):
            C4[n] = (
                np.nanmean((x[0 : -int(i)] - x[int(i) :]) ** 4)
                - 3 * np.nanmean((x[0 : -int(i)] - x[int(i) :]) ** 2) ** 2
            )

        return C4

    def _C4(self) -> dict:
        """Compute the forth cumulant along the 3 axis of the trajectory."""

        self.C4 = {}
        self.C4_t = np.logspace(
            np.log(1) / np.log(10),
            np.log(len(self.Data) / 10) / np.log(10),
            num=self.MSD_bins,
        )
        self.C4_t = np.unique(self.C4_t.astype(int))
        for n, i in enumerate(self.axis):
            if self.verbose:
                print("Computing C4 on " + i)
            self.C4[i] = self._C4_1D(self.C4_t, self.Data[:, n])

        return self.C4

    def Long_time_MSD(self):
        """Compute the mean value of the MSD plateau, t_LMSD gives the range,
        in seconds between which the average is computed."""

        I1 = np.argwhere(self.MSD_t * self.Data.dt > self.t_LMSD[0])[0]
        I2 = np.argwhere(self.MSD_t * self.Data.dt > self.t_LMSD[1])[0]

        self.plateau = np.nanmean(self.MSD["z"][int(I1) : int(I2)])

    def Short_time_PDF(self):

        """Compute the PDF of discplacement along both x, y and z axis.
        A PDF is computed for each available time step in the range given by t_sPDF"""

        I1 = int(self.t_sPDF[0] * self.Data.fps)
        if I1 == 0:
            I1 += 1
        I2 = int(self.t_sPDF[1] * self.Data.fps)

        short_time_PDF_Dz = {}
        short_time_PDF_Dx = {}
        short_time_PDF_Dy = {}
        I = np.arange(I1, I2).astype(int)

        for i in I:

            Dezs = self.Data.z[:-i] - self.Data.z[i:]
            Dexs = self.Data.x[:-i] - self.Data.x[i:]
            Deys = self.Data.y[:-i] - self.Data.y[i:]

            hist_x, bins_center_x = self._pdf(
                Dexs[~np.isnan(Dexs)], bins=self.SPDF_bins
            )
            hist_x = hist_x / np.trapz(hist_x, bins_center_x)

            hist_y, bins_center_y = self._pdf(
                Dexs[~np.isnan(Deys)], bins=self.SPDF_bins
            )
            hist_y = hist_y / np.trapz(hist_y, bins_center_y)

            hist_z, bins_center_z = self._pdf(
                Dezs[~np.isnan(Dezs)], bins=self.SPDF_bins
            )
            hist_z = hist_z / np.trapz(hist_z, bins_center_z)

            short_time_PDF_Dx[str(i)] = {
                "bin_center": bins_center_x,
                "PDF": hist_x,
                "std": np.std(Dexs),
            }

            short_time_PDF_Dy[str(i)] = {
                "bin_center": bins_center_y,
                "PDF": hist_y,
                "std": np.std(Dexs),
            }

            short_time_PDF_Dz[str(i)] = {
                "bin_center": bins_center_z,
                "PDF": hist_z,
                "std": np.std(Dezs),
            }

        self.short_time_PDF_Dx = short_time_PDF_Dx
        self.short_time_PDF_Dy = short_time_PDF_Dy
        self.short_time_PDF_Dz = short_time_PDF_Dz

    def Large_Dt_PDF(self):
        """Compute the large time displacement pdf"""
        t_Lpdf = self.t_Lpdf
        fps = self.Data.fps
        I = np.arange(t_Lpdf[0] * fps, t_Lpdf[1] * fps).astype(int)

        hists = np.zeros((self.LPDF_bins, len(I)))
        bins_centers = np.zeros((self.LPDF_bins, len(I)))

        for n, i in enumerate(I):

            Dezs = self.Data.z[:-i] - self.Data.z[i:]
            hist, bins_center = self._pdf(Dezs[~np.isnan(Dezs)], bins=self.LPDF_bins)

            hists[:, n] = hist
            bins_centers[:, n] = bins_center

        self.pdf_long_t = np.mean(hists, axis=1)
        self.bins_centers_long_t = np.mean(bins_centers, axis=1)
        self.err_long_t = np.std(hists, axis=1)
        self.err_bins_centers = np.std(bins_centers, axis=1)

    def local_diffusion(self):
        """Computing the local diffusion using the Frishman and Ronceray method."""
        pos = np.ones((len(self.Data.x), 3))

        pos[:, 0] = self.Data.x
        pos[:, 1] = self.Data.y
        pos[:, 2] = self.Data.z

        self.Dx, self.Dy, self.Dz, self.z_D = Compute_diffusion(
            pos, self.Data.dt, *self.range_D, N=self.N_local_D, ordre=self.ordre_D
        )

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

                for i in self.axis:
                    ax.loglog(
                        self.MSD_t * self.Data.dt, self.MSD[i], "o", label="$" + i + "$"
                    )
                    ax.plot(
                        self.MSD_t[-30:] * self.Data.dt,
                        [self.plateau] * 30,
                        color="k",
                        linewidth=5,
                        alpha=0.5,
                    )
                ax.legend(frameon=False)
                ax.set(xlabel="$t$ (s)", ylabel="MSD (m$^2$)")
            case "pdf":

                ax.semilogy(self.z_pdf_z, self.pdf_z, "o")
                ax.set(xlabel="$z$ ($\mu$m)", ylabel="PDF (m$^{-1}$)")

            case "f_eq":
                ax.semilogx(self.z_F_eq, self.F_eq * 1e15, "o")

                ax.set(xlabel="$z$ (m)", ylabel="$F$ (fN)")
            case "c4":
                for i in self.axis:
                    ax.loglog(
                        self.C4_t * self.Data.dt, self.C4[i], "o", label="$" + i + "$"
                    )
                ax.legend(frameon=False)
                ax.set(xlabel="$t$ (s)", ylabel="C4 (m$^4$)")

            case "short_time_pdf_x":

                I1 = int(self.t_sPDF[0] * self.Data.fps)
                if I1 == 0:
                    I1 += 1
                I2 = int(self.t_sPDF[1] * self.Data.fps)
                I = np.arange(I1, I2)

                for i in I:
                    PDF = self.short_time_PDF_Dx[str(i)]

                    ax.semilogy(
                        PDF["bin_center"],
                        PDF["PDF"],
                        "o",
                        label="$\Delta t  = " + str(i / self.Data.fps) + "$ s",
                    )
                ax.legend(frameon=False)
                ax.set(xlabel="$\Delta x$ (m)", ylabel="PDF (m$^-1$)")
            case "short_time_pdf_z":

                I1 = int(self.t_sPDF[0] * self.Data.fps)
                if I1 == 0:
                    I1 += 1
                I2 = int(self.t_sPDF[1] * self.Data.fps)
                I = np.arange(I1, I2)

                for i in I:
                    PDF = self.short_time_PDF_Dz[str(i)]

                    ax.semilogy(
                        PDF["bin_center"],
                        PDF["PDF"],
                        "o",
                        label="$\Delta t  = " + str(i / self.Data.fps) + "$ s",
                    )
                ax.legend(frameon=False)
                ax.set(xlabel="$\Delta z$ (m)", ylabel="PDF (m$^-1$)")
            case "long_time_pdf":

                ax.errorbar(
                    self.bins_centers_long_t,
                    self.pdf_long_t,
                    yerr=self.err_long_t,
                    fmt="o",
                    barsabove=False,
                    linestyle="",
                    ecolor="k",
                    capsize=4,
                )
                ax.semilogy()
                ax.set(xlabel="$\Delta z$ (m)", ylabel="PDF (m$^-1$)")
            case "local_d":
                z_D = self.z_D
                Dx = self.Dx
                Dy = self.Dy
                Dz = self.Dz
                Do = self.Do

                ax.loglog(z_D, Dz / Do, "o", label=r"$D_\bot$")
                # ax.plot(z_D, Dx / Do,"o", label="Dx")
                # ax.plot(z_D, Dy / Do,"o" label="Dy")
                ax.plot(z_D, (Dx + Dy) / 2 / Do, "o", label=r"D_\\parallel")
                ax.legend(frameon=False)

                ax.set(xlabel="$z$ (m)", ylabel="$D/ D_0$ (m$^2$s$^-1$)")
        if needs_plot:
            plt.show()

    def general_plot(self):
        plt.ioff()
        plt.figure(figsize=(10, 10))
        plt.ion()

        ax1 = plt.subplot(421)
        self.plot("msd", ax1)

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
