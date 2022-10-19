from copy import copy
import numpy as np


class Dedrift:
    """Class use to deal with the dedrift of the trajectory"""

    def __init__(self, traj: np.ndarray, method: str = "min_z", **kwargs):
        """
        You can choose to create new methods to dedrift the trajectory,
        in order to do it. You can create a new method and add it to the switch below.
        Additional arguments can be passed through kwargs.
        """
        self.method = method
        self.traj = traj
        available_method = ["min_z", "mean_z", "global_min"]
        if not (self.method in available_method):
            raise ValueError(
                "Please select an available method :" + str(available_method)
            )

        self.unchanged_traj = copy(traj)

        ### If you want to add a dedrift method, you can add it in the switch.
        match self.method:
            case "min_z":
                params = ["window"]
                self._check_params(kwargs, params, self.method)
                self.min_z(kwargs["window"])

            case "mean_z":
                params = ["window"]
                self._check_params(kwargs, params, self.method)
                self.min_z(kwargs["window"])

            case "global_min":
                self.global_min()

    def _movmean(self, datas: np.ndarray, window: int):

        """Moving average function"""
        result = np.empty_like(datas)
        start_pt = 0
        end_pt = int(np.ceil(window / 2))

        for i in range(len(datas)):
            if i < int(np.ceil(window / 2)):
                start_pt = 0
            if i > len(datas) - int(np.ceil(window / 2)):
                end_pt = len(datas)
            result[i] = np.mean(datas[start_pt:end_pt])
            start_pt += 1
            end_pt += 1

        return result

    def _movmin(self, datas: np.ndarray, window: int):

        """Moving minimum function"""

        result = np.empty_like(datas)
        start_pt = 0
        end_pt = int(np.ceil(window / 2))

        for i in range(len(datas)):
            if i < int(np.ceil(window / 2)):
                start_pt = 0
            if i > len(datas) - int(np.ceil(window / 2)):
                end_pt = len(datas)
            result[i] = np.min(datas[start_pt:end_pt])
            start_pt += 1
            end_pt += 1

        return result

    def _check_params(self, kwargs: dict, params: list, method: str):
        for i in params:
            if not (i in kwargs):
                raise ValueError(
                    "Using the {} method requieres you to set the parameter {}  (dedrift(traj, {})).".format(
                        method, i, i
                    )
                )

    def min_z(self, window: int):
        """Simple dedrifting strategy, removing a moving minimum to the z trajectory."""
        self.traj[:, 2] -= self._movmin(self.traj[:, 2], window)

    def mean_z(self, window: int):
        """Simple dedrifting strategy, removing a moving average to the z trajectory, and set the minimum to zero"""
        self.traj[:, 2] -= self._movmin(self.traj[:, 2], window)
        self.traj[:, 2] -= np.min(self.traj[:, 2])

    def global_min(self):
        """No dedrift is done, the minimum is just set to zero"""
        self.traj[:, 2] -= np.min(self.traj[:, 2])
