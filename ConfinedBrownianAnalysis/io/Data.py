import numpy as np
from typing import Union
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from ConfinedBrownianAnalysis.Analyse import Dedrift


class Data(np.ndarray):
    def __new__(cls, file: str, fps: int = 100, cutoff: int = -1):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        cls.file = loadmat(file)
        initial_array = np.array([cls.file["x"], cls.file["y"], cls.file["z"]])[
            :, 0, :cutoff
        ]
        obj = np.asarray(initial_array).view(cls)

        obj = obj.transpose()

        return obj

    def __init__(
        self, file: str, fps: int = 100, cutoff: int = -1, dedrift_method="min_z"
    ):

        self.fps = fps
        self.dt = 1 / fps
        # Adding the x, y and z aatribute for easier use.
        self.x = np.squeeze(np.array(self.file["x"]))[:cutoff]
        self.y = np.squeeze(np.array(self.file["y"]))[:cutoff]
        self.z = np.squeeze(np.array(self.file["z"]))[:cutoff]
        self._x = np.squeeze(np.array(self.file["x"]))[:cutoff]
        self._y = np.squeeze(np.array(self.file["y"]))[:cutoff]
        self._z = np.squeeze(np.array(self.file["z"]))[:cutoff]
        self.initial_array = self.__array__

        # trajectory time
        self.time = np.arange(len(self.x)) / fps
        self._time = np.arange(len(self.x)) / fps

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):

            self.x = self._x.__getitem__(subscript)
            self.y = self._y.__getitem__(subscript)
            self.z = self._z.__getitem__(subscript)
            self.time = self._time.__getitem__(subscript)
            self.__array__ = self.initial_array
            return super().__getitem__(subscript)
        return super().__getitem__(subscript)

    def __array_finalize__(self, obj):
        """
        The behaviour when terminating the class, this also done when slicing or
        multiplying the array. For example, it permits to keep the attribute when slincing
        the Data class.
        """
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        # self.info = getattr(obj, "info", None)
        self.fps = getattr(obj, "fps", None)
        self.x = getattr(obj, "x", None)
        self.y = getattr(obj, "y", None)
        self.z = getattr(obj, "z", None)
        self._x = getattr(obj, "x", None)
        self._y = getattr(obj, "y", None)
        self._z = getattr(obj, "z", None)
        self.time = getattr(obj, "time", None)
        self._time = getattr(obj, "time", None)
        self.initial_array = getattr(obj, "initial_array", None)
        self.dt = getattr(obj, "dt", None)

    def plot_3D(self, N: int = 20, N_c: int = 500):
        """
        Plot the trajectory in 3D, using N chunks of N_c points with a gradient of color indicating the time.
        """
        plt.ioff()
        fig = plt.figure()
        plt.ion()
        ax = plt.axes(projection="3d")

        cmap = plt.get_cmap("jet")

        for i in range(N - 1):
            ax.plot(
                self.x[i * N_c : i * N_c + N_c],
                self.y[i * N_c : i * N_c + N_c],
                self.z[i * N_c : i * N_c + N_c],
                color=plt.cm.jet(1 * i / N),
            )

        ax = plt.gca()
        ax.ticklabel_format(style="sci")

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.xlabel("$x$ ($\mathrm{\mu m}$)")
        plt.ylabel("$y$ ($\mathrm{\mu m}$)")
        ax.set_zlabel("$z$ ($\mathrm{\mu m}$)")

        ticks_c = []
        for i in np.linspace(0, 1, 5):
            ticks_c.append("{:.0f}".format(N * N_c * i / self.fps / 60))
        cbar = plt.colorbar(
            sm,
            ticks=np.linspace(0, 1, 5),
            format="%.1f",
            shrink=0.5,
            orientation="vertical",
            pad=0.2,
        )
        cbar.set_ticklabels(ticks_c)
        cbar.set_label("$t ~ \mathrm{(s)}$")
        plt.show()

    def plot_1D(self, axis: Union[str, int]):

        if axis not in ["x", "y", "z", 0, 1, 2]:
            raise ValueError(
                "Please choose an axis in ['x', 'y', 'z'] or the numeric value [0, 1, 2]"
            )

        axis_dict = {"x": 0, "y": 1, "z": 2}
        axis_dict_inverted = {"0": "x", "1": "y", "2": "z"}
        if type(axis) == str:
            axis = axis_dict[axis]

        plt.ioff()
        fig = plt.figure()
        plt.ion()

        plt.plot(self.time, self[:, axis])

        plt.xlabel("$t$ ($\mathrm{s}$)")
        plt.ylabel("$" + axis_dict_inverted[str(axis)] + "$" + "($~\mathrm{\mu m}$)")
        plt.tight_layout()
        plt.show()

    def dedrift(self, method: str = "min_z", **kwargs):
        ded = Dedrift(self, method, **kwargs)
        self = ded.traj
        self.x = self[:, 0]
        self.y = self[:, 1]
        self.z = self[:, 2]

    def clear(self):
        plt.close("all")
