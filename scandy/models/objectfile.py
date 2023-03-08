import numpy as np
from scipy.ndimage.measurements import center_of_mass


class ObjectFile:
    """
    All object files that accumulate evidence for the Drift Diffusion Model are Object Files.
    The Nomenclature is inspired by the idea of Kahnemann et al. ~1980.
    Class is initialized by the dictionary created by load_object_parameters
    TODO: nice represent function!
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
        self.ior = 0  # used to update IOR
        self.inhibition = 0  # effective IOR value used for the decision
        # self.inobj_ior = 0 --> scalar model parameter

        # avoid calculations if object does not appear in a frame using this list
        self.appears_in = [
            np.any(self.object_maps[f]) for f in range(self.object_maps.shape[0])
        ]
        if not self.ground:
            position = [
                self._get_position(f, apear) for f, apear in enumerate(self.appears_in)
            ]
            self.shift = [0] + [
                position[f] - position[f - 1] for f in range(1, len(position))
            ]
        # not needed: else: self.position = [np.array([0,0],dtype=int) for i in range(len(self.appears_in))]

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
        self.ior = 0
        self.inhibition = 0
        self.decision_variable = 0

    def _get_position(self, frame, appearence):
        """
        Calculate the middle point of the object mask in a given frame, only used for calculating the shift
        """
        if appearence:
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

    def update_ior(
        self, ior_decay, ior_inobj
    ):  # VARIANT: for "maxIORinobj", ior_inobj):
        """
        This method accounts for forgetting. IOR is increasing up to 1 while an object is foveated.
        When it is not foveated, this value goes back to 0 over time, with the update -1/ior_decay per frame.
        TODO?? Instead of growing to 1 (taking up in memory), we set it to 1 as soon as it's foveated!
        """
        if self.foveated:
            self.ior = 1
            if self.ground:
                self.inhibition = 0.0
            else:
                self.inhibition = ior_inobj
            # self.ior = (
            #     1.0  # VARIANT: for "maxIORinobj": * ior_inobj  # min(self.ior, 1)
            # )
        else:
            if self.ior > 0:
                self.ior -= 1.0 / ior_decay
                self.ior = max(self.ior, 0)
            self.inhibition = self.ior * 1.0

    def calc_decision_variable(self, frame, sensitivitymap, featuremap, ddm_sig):
        """
        This function is the heart of the decision making model!
        It accumulates the evidence to look at this object in the given frame
        This depends on the features within the object in the given frame (standard: Molin saliency),
        the size of the object mask (prevously free parameter gamma, now scaled with log),
        and the current gaze position (att_dva).
        For an intra object saccade, the attention spreads uniformly over the objects (intraobjatt)
        """
        # only update evidence according to scene if the object is visible
        if self.appears_in[frame]:
            mu = (
                np.sum((self.object_maps[frame] * sensitivitymap * featuremap))
                / self.pxsize[frame]
                * np.log(self.pxsize[frame])  # np.sqrt(self.pxsize[frame])  #
                * (1 - self.inhibition)
            )

            ### OLD: different IOR treatment, now this is dealt with in update_ior!
            # # attention spread is different for within- or between-object saccades!
            # if self.foveated:
            #     # attention on the perceptual ground spreads like a Gaussian
            #     if self.ground:
            #         mu = (
            #             np.sum((self.object_maps[frame] * sensitivitymap * featuremap))
            #             / self.pxsize[frame]
            #             * np.log(
            #                 self.pxsize[frame]
            #             )  # np.sqrt(self.pxsize[frame])  # self.pxsize[frame] * np.log(self.pxsize[frame])
            #         )  # * (1 - self.ior * ior_inobj) before --> no IOR in background
            #     else:
            #         # CHANGED INTERPRETATION: Attention on objects spreads uniform with 1 across whole object!
            #         # CHANGED 23.4.21: scaling used to be pxsize * np.log(pxsize),
            #         #  --> but one is more likely to make saccade within an object if it's bigger!
            #         #  --> sqrt used here: https://jov.arvojournals.org/article.aspx?articleid=2193943#88379220
            #         mu = (
            #             np.sum((self.object_maps[frame] * featuremap))
            #             / self.pxsize[frame]
            #             * np.log(self.pxsize[frame])  # np.sqrt(self.pxsize[frame])
            #             * (1 - self.ior * ior_inobj)
            #         )
            # else:
            #     # accumulate evidence based on feature map and attention
            #     mu = (
            #         np.sum((self.object_maps[frame] * sensitivitymap * featuremap))
            #         / self.pxsize[frame]
            #         * np.log(self.pxsize[frame])  # np.sqrt(self.pxsize[frame])  #
            #         * (1 - self.ior)
            #     )

            self.decision_variable += mu + np.random.normal(0, ddm_sig)

        # if object is not visible in one frame, decrease the evidence by 10% (--> exponentially to zero)
        else:
            self.decision_variable *= 0.9

        return self.decision_variable
