import numpy as np
from scipy.ndimage.measurements import center_of_mass


class ObjectFile:
    """
    Object files are representations, `, within which successive states of an
    object are linked and integrated` (Kahnemann et al. 1992).
    Class is initialized for each video within load_videodata using the segmentation masks.
    """

    def __init__(self, obj_id, object_masks):
        self.id = obj_id
        if obj_id == 0:
            self.ground = True
            self.name = "Ground"
        else:
            self.ground = False
            self.name = f"Object {obj_id}"
        # create a binary numpy.ndarray with true for every pixel with the object
        self.object_maps = object_masks == obj_id
        # initialize attributes that are relevant for the evidence accumulation
        self.foveated = False
        self.decision_variable = 0
        self._ior_tempmem = 0  # used to update IOR
        self.ior = 0  # effective IOR value used for the decision

        # avoid calculations if object does not appear in a frame using this list
        self.appears_in = [
            np.any(self.object_maps[f]) for f in range(self.object_maps.shape[0])
        ]
        # to avoid recalculations, we calculate the shift of the object in each frame
        #   but only if the object is not the background
        if not self.ground:
            position = [
                self._get_position(f, apear) for f, apear in enumerate(self.appears_in)
            ]
            self.shift = [0] + [
                position[f] - position[f - 1] for f in range(1, len(position))
            ]

        # it is well established that object size plays a role and e.g. Nuthmann2020 scale it with the log
        # we therefore calculate the average features (TERM / pxsize) and multiply it with np.log(pxsize)
        self.pxsize = [
            0 if not apear else np.sum(self.object_maps[f])
            for f, apear in enumerate(self.appears_in)
        ]

    def set_initial_state(self):
        """
        Convenience function that resets the initial state.
        Always run this before a trial to make sure nothing is caried over!
        """
        self.foveated = False
        self._ior_tempmem = 0
        self.ior = 0
        self.decision_variable = 0

    def _get_position(self, frame, appears_in_frame):
        """
        Calculate the middle point of the object mask in a given frame, only used for calculating the shift
        """
        if appears_in_frame:
            return np.array(center_of_mass(self.object_maps[frame]), dtype=int)
        else:
            return np.array([0, 0], dtype=int)

    def update_foveation(self, frame, gaze_loc):
        """
        We need to know which object is currently foveated, this could change without saccade events due to the drift
        Therefore, this update has to run for each object for each frame.
        CAVEAT: This attribute has no tolerance, different from the evaluation!
        """
        # gaze_loc = [y,x] --> this is correct!
        self.foveated = (
            self.appears_in[frame] and self.object_maps[frame, gaze_loc[0], gaze_loc[1]]
        )

    def update_ior(self, model_params):
        """
        This method accounts for the scanpath history. If the object is foveated, the IOR is set to 1.
        Objects (except background) are inhibited during foveation with model_params["ior_inobj"].
        When it is not foveated, this value goes back to 0 over time, with the update -1/ior_decay per frame.
        """
        if self.foveated:
            # if not foveated anymore, it decreases starting from 1
            self._ior_tempmem = 1
            if self.ground:
                self.ior = 0.0
            else:
                self.ior = model_params["ior_inobj"]
        else:
            # if not foveated anymore, decrease IOR (only if larger than 0)
            if self._ior_tempmem > 0:
                self._ior_tempmem -= 1.0 / model_params["ior_decay"]
                self._ior_tempmem = max(self._ior_tempmem, 0)
            self.ior = self._ior_tempmem * 1.0
        return self.ior

    def update_evidence(self, frame, feature_map, sens_map, model_params):
        """
        Accumulates evidence in favor of moving the eyes towards the object.

        Drift diffusion process where the drift rate is proportional to the
        features and sensitivity within the object mask and how strongly the
        object is inhibited.

        :param frame: current frame
        :type frame: str
        :param feature_map: Feature map of the current frame, loaded in modul I
        :type feature_map: np.ndarray
        :param sens_map: Sensitivity map of the current frame, updated in modul II
        :type sens_map: np.ndarray
        :param model_params: Model parameters, needed for the DDM noise
        :type model_params: dict
        :return: Updated decision variable for the object
        :rtype: float
        """
        # only update evidence according to scene if the object is visible
        if self.appears_in[frame]:
            mu = (
                np.sum((self.object_maps[frame] * sens_map * feature_map))
                / self.pxsize[frame]
                * np.log(self.pxsize[frame])
                * (1 - self.ior)
            )

            self.decision_variable += mu + np.random.normal(0, model_params["ddm_sig"])

        # if object is not visible in one frame, decrease the evidence by 10% (--> exponentially to zero)
        else:
            self.decision_variable *= 0.9

        return self.decision_variable
