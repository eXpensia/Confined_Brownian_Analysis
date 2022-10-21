import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz

class Model:
    def __init__(
        self,
        B=4,
        lD=70e-9,
        a=1.5e-6,
        Drho=50,
        eta=1e-3,
        z_0=0,
        b=0,
        noise_lvl_MSD=0,
    ):

        self.B = B
        self.lD = lD
        self.a = a
        self.Drho = Drho
        self.eta = eta
        self.z_0 = z_0
        self.b = b
        self.noise_lvl_MSD = noise_lvl_MSD
        self.z_th = np.linspace(1e-9, 16e-6, 10000)
        self.z_D_th = self.z_th - self.b
        self.lB = 4e-21 / (4 / 3 * np.pi * Drho * 9.81 * a**3)
        self.D0 = 4e-21 / (6 * np.pi * eta * a)

    def P_0(self, z, normalize=True):

        P = np.exp(-self.B * np.exp(-z / self.lD) - z / self.lB)
        P[z < 0] = 0

        if normalize:
            A = np.trapz(P, z)
            return P / A

        return P

    def P_0_off(self, z):

        z = z - self.z_0
        P = np.exp(-self.B * np.exp(-z / self.lD) - z / self.lB)

        if type(z) == float:
            if z < self.z_0:
                P = 0
            return P

        P[z < self.z_0] = 0
        A = np.trapz(P, z)
        return P / A

    def Dz(self, z):
        z = z + self.b
        return (6 * z**2 + 2 * self.a * z) / (
            6 * z**2 + 9 * self.a * z + 2 * self.a**2
        )

    def Dx(self, z):
        z = z + self.b  # + due to usual slippage convention
        xi = self.a / (z + self.a)
        return 1 - 9 / 16 * xi + 1 / 8 * xi**3 - 45 / 256 * xi**4 - 1 / 16 * xi**5

    ## Modified diffusion coefficient function to take into account z_0

    def Dz_off(self, z):
        z = z + self.b  # + due to usual slippage convention
        z = z - self.z_0
        return (6 * z**2 + 2 * self.a * z) / (
            6 * z**2 + 9 * self.a * z + 2 * self.a**2
        )

    def Dx_off(self, z):
        z = z + self.b  # + due to usual slippage convention
        z = z - self.z_0
        xi = self.a / (z + self.a)
        return 1 - 9 / 16 * xi + 1 / 8 * xi**3 - 45 / 256 * xi**4 - 1 / 16 * xi**5

    def P_D_short_time(self, Dz, Dt, axis="x"):

        z = self.z_th

        if axis == "x":
            D = self.Dx(z) * self.D0
        else:
            D = self.Dz(z) * self.D0

        P_D = D * self.P_0(z)
        P_D /= np.trapz(P_D, z)

        P = lambda Dz: np.trapz(
            P_D / np.sqrt(4 * np.pi * D * Dt) * np.exp(-(Dz**2) / (4 * D * Dt)), D
        )

        P_Dz = np.array([P(i) for i in Dz])
        return P_Dz / np.trapz(P_Dz, Dz)

    def mean_D(self, axis="x"):
        z = self.z_th

        if axis == "x":
            D = self.Dx(z) * self.D0
        else:
            D = self.Dz(z) * self.D0

        return np.trapz(self.P_0(z) * D, z)

    def long_time_pdf(self, Dz):

        z = self.z_th

        def P_long(Dz):
            dp = self.P_0(z, normalize=False) * self.P_0(z + Dz, normalize=False)
            P = trapz(dp, z)
            return P

        if type(Dz) == float:
            return P_long(Dz)

        pdf = np.array([P_long(i) for i in Dz])
        return pdf / trapz(pdf, Dz)

    def plateau_MSD(self):

        dz = np.linspace(-5e-6, 5e-6, 10000)
        P_Dz = self.long_time_pdf(dz)

        return np.trapz(dz**2 * P_Dz, dz)

    def plateau_C4(self):

        dz = np.linspace(-5e-6, 5e-6, 10000)
        P_Dz = self.long_time_pdf(dz)

        return np.trapz(dz**4 * P_Dz, dz) ** 2 - 3 * np.trapz(dz**2 * P_Dz, dz) ** 2

    def MSD_short_time(
        self,
        t,
        axis="x",
    ):
        return 2 * self.mean_D(axis=axis) * t

    def C4_short_time(self, t, axis="x"):
        z = self.z_th

        if axis == "x":
            mean_D2 = np.trapz(self.Dx(z) ** 2 * self.P_0(z), z)
        else:
            mean_D2 = np.trapz(self.Dz(z) ** 2 * self.P_0(z), z)

        return 12 * (mean_D2 - self.mean_D(axis=axis) ** 2) * t**2

    def Conservative_Force(self, z):
        z = z - self.z_0

        return 4e-21 * (self.B / (self.lD) * np.exp(-z / (self.lD)) - 1 / (self.lB))
