import numpy as np
from typing import Union
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from ConfinedBrownianAnalysis.Analyse import Dedrift
import pickle

def Load(filename):
    """ Simple function to load a pickled object """
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

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
        del cls.file
        return obj

    def __init__(
        self, file: str, fps: int = 100, cutoff: int = -1, dedrift_method="min_z"
    ):

        self.fps = fps
        self.dt = 1 / fps


        # trajectory time
        self.time = np.arange(len(self.x)) / fps
        self._time = np.arange(len(self.x)) / fps

    @property
    def x(self):
        return self.__array__()[:,0]

    @property
    def y(self):
        return self.__array__()[:,1]

    @property
    def z(self):
        return self.__array__()[:,2]




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


    def clear(self):
        plt.close("all")




    # Handeling the picke and slice of the array

    def __getitem__(self, subscript):
        """
        Rewrite the slicing function __getitem__ so that it also slice
        self.time at the same time.
        """
        if isinstance(subscript, slice):

            self.time = self._time.__getitem__(subscript)
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

        self.fps = getattr(obj, "fps", None)
        self.time = getattr(obj, "time", None)
        self._time = getattr(obj, "_time", None)
        self.dt = getattr(obj, "dt", None)

    def __reduce__(self):
        """
        np.ndarray uses __reduce__ to pickle itself, so we need to rewrite them in order to save new attributes.
        https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        """
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()

        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fps,self.time, self._time, self.dt)
        # print(new_state)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.fps = state[-4]  # Set the info attribute
        self.time = state[-3]
        self._time = state[-2]
        self.dt = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(Data, self).__setstate__(state[0:-4])


    ### Saving with pickle ###

    def save(self, filename):
        with open(filename, "wb") as handle:
            b = pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
