# from . import parameterDict as par
import os
import numpy as np
import matplotlib.pyplot as plt
import gif
import flow_vis as fv

from ..model import Model
from ..objectfile import ObjectFile
from ...utils import functions as uf
from neurolib.utils.collections import dotdict


class ObjectModel(Model):
    """
    Basic object-based model: potential saccade targets are object files.
    """

    name = "obj"
    description = "Basic object-based model with calculations within object files."

    def __init__(self, Dataset, params=None, preload_res_df=None):
        """
        Initialize the model. IOR and decision variables for each objec file.

        :param Dataset: Data for which the model will be run
        :type Dataset: Dataset class
        :param params: Model parameters, loaded internally, defaults to None
        :type params: dict, optional
        :param preload_res_df: Only if computations were already done, defaults to None
        :type preload_res_df: DataFrame, optional
        """
        self.Dataset = Dataset
        self.params = params
        # load default parameters if none were given
        if self.params is None:
            self.params = self.load_default_modelparams()

        # Initialize base class Model
        super().__init__(
            Dataset=Dataset,
            params=self.params,
            preload_res_df=preload_res_df,
        )

        # Attributes that are updated when loading a video
        self.video_data = {}
        # Attributes that are updated when running the model for a single video
        self._scanpath = []
        self._f_sac = []
        self._gaze_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._new_target = None
        # self._prev_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._current_frame = 0
        # sensitivity and features are given as maps
        self._feature_map = np.zeros((self.Dataset.VID_SIZE_Y, self.Dataset.VID_SIZE_X))
        self._sens_map = np.zeros_like(self._feature_map)
        # object based value arrays (number of objects set correctly in reinit)
        self._ior_objs = np.zeros(5)
        self._decision_objs = np.zeros(5)
        # Stored maps / objectvals for a single run for visualization
        self._all_dvs = []
        self._all_iors = []
        self._all_sens = []
        # self._all_features = [] # only if features are modified, otherwise save memory

    def load_default_modelparams(self):
        """
        Defines the model specifications, most importantly the feature map (defaults
        to low-level features with anisotropic center bias).
        Free parameters that should be adapted are ior_decay, ior_inobj, att_dva,
        ddm_thres, ddm_sig.

        :return: Parameter dictionary
        :rtype: dict
        """
        par = dotdict({})

        # random seed for a scanpath simulation, is set in the model.run function
        par.rs = None
        # starting position, default is center
        par.startpos = np.array(
            [self.Dataset.VID_SIZE_Y // 2, self.Dataset.VID_SIZE_X // 2]
        )
        par.ior_decay = 1.0 * self.Dataset.FPS
        # Inhibition of currently foveated object
        par.ior_inobj = 0.5  # range: [0,1]
        # Spread of Gaussian visual sensitivity around the gaze point
        par.att_dva = 6.0
        # Drift-diffusion model
        par.ddm_thres = 0.3
        par.ddm_sig = 0.01
        # par.ddm_reset = 0  # add this param if update_decision is changed!
        # Oculomotor drift
        par.drift_sig = self.Dataset.DVA_TO_PX / 8.0
        # under normal circumstances, the model does not need flow information
        par.use_flow = False
        # Objects are definitely used in this model
        par.use_objects = True
        par.use_objectfiles = True
        # Choose the used feature map and whether a center bias should be used
        par.centerbias = "anisotropic_default"
        par.featuretype = "molin"
        assert os.path.isdir(
            f"{self.Dataset.featuremaps}{par.featuretype}/"
        ), f"No stored features at {self.Dataset.featuremaps}{par.featuretype}/"
        # Determines how verbose the return dictionary is, detailed info only for single runs
        par.sglrun_return = True

        return par

    def reinit_for_sgl_run(self):
        """
        Reinitialize the model such that all parameters which might be updated
        in a simulation run are back in their initial state.
        """
        for obj in self.video_data["object_list"]:
            obj.set_initial_state()
        self._scanpath = []
        self._f_sac = []
        self._current_frame = 0
        self._gaze_loc = np.zeros(2, int)
        self._prev_loc = np.zeros(2, int)  # not used, but potentially for IOR
        self._new_target = None
        self._feature_map.fill(0.0)
        self._sens_map.fill(0.0)
        self._ior_objs = np.zeros(len(self.video_data["object_list"]))
        self._decision_objs = np.zeros(len(self.video_data["object_list"]))
        # Stored maps / object values for a single run for visualization
        self._all_dvs = []
        self._all_iors = []
        self._all_sens = []

    def update_features(self):
        """
        Modul (I): Scene features
        Current frame of the features, loaded as specified in the model parameters.

        TRYOUT: could use a time-dependent center bias
        """
        self._feature_map = self.video_data["feature_maps"][self._current_frame]

    def update_sensitivity(self):
        """
        Modul (II): Visual sensitivity
        Visual sensitivity map, Gaussian spread around the gaze point, with the
        currently foveated object (if not background) having sensitivity 1.

        TRYOUT: increase sensitivity for objects that are close to the decision
        threshold (i.e. presaccadic attention)
        """
        assert self.params is not None, "Model parameters not loaded"

        for obj in self.video_data["object_list"]:
            obj.update_foveation(self._current_frame, self._gaze_loc)
        # Gaussian sensitivity spread around the gaze point
        gaze_gaussian = uf.gaussian_2d(
            self._gaze_loc[1],
            self._gaze_loc[0],
            self.Dataset.VID_SIZE_X,
            self.Dataset.VID_SIZE_Y,
            self.params["att_dva"] * self.Dataset.DVA_TO_PX,
        )
        # set sensitivity to 1 for currently foveated object (not background)
        for obj in self.video_data["object_list"][1:]:
            if obj.foveated:
                gaze_gaussian[obj.object_maps[self._current_frame]] = 1.0

        self._sens_map = gaze_gaussian
        if self.params["sglrun_return"]:
            self._all_sens.append(self._sens_map.copy())

    def update_ior(self):
        """
        Modul (III): Scanpath history
        Object-based inhibition of return, updated for all objects in the scene.

        TRYOUT: make location-based map, as proposed by Krüger et al. (2013)
        """
        assert self.params is not None, "Model parameters not loaded"
        for obj_id, obj in enumerate(self.video_data["object_list"]):
            self._ior_objs[obj_id] = obj.update_ior(self.params)
        if self.params["sglrun_return"]:
            self._all_iors.append(self._ior_objs.copy())

    def update_decision(self):
        """
        Modul (IV): Decision making
        Update the decision variable for all objects in the scene, and check if
        any object crossed the decision threshold.

        TRYOUT: Instead of accumulating in DDM, make saccade if the update in
        a single timestep is above a threshold.
        """
        assert self.params is not None, "Model parameters not loaded"
        for obj_id, obj in enumerate(self.video_data["object_list"]):
            self._decision_objs[obj_id] = obj.update_evidence(
                self._current_frame, self._feature_map, self._sens_map, self.params
            )

        if np.max(self._decision_objs) > self.params["ddm_thres"]:
            # find the object that crossed the threshold ==> becomes the target
            self._new_target = np.argmax(self._decision_objs)
            # reset decision variables for all objects
            for obj in self.video_data["object_list"]:
                # TRYOUT: We could set to self.params["ddm_reset"] instead of 0
                obj.decision_variable = 0.0
        else:
            self._new_target = None
        if self.params["sglrun_return"]:
            self._all_dvs.append(self._decision_objs.copy())

    def update_gaze(self):
        """
        Module(V): Gaze update
        Either saccade to a position within the target object, or follow the object
        doing smooth pursuit and fixational eye movements.

        TRYOUT: Landing position could also be more likely towards the object
        center (but not if the objects are humans, then faces are too important)
        """
        assert self.params is not None, "Model parameters not loaded"
        if self._new_target is not None:
            # probability of new location within target object depends on features
            probmap = (
                self.video_data["object_list"][self._new_target].object_maps[
                    self._current_frame
                ]
                * self.video_data["feature_maps"][self._current_frame]
                * self._sens_map
            )
            probmapsum = np.sum(probmap)
            # draw new location from probability map
            # if features are all zero in the target (rare!) choose uniformly
            if np.isclose(probmapsum, 0):
                self._gaze_loc = np.array(
                    np.unravel_index(
                        np.random.choice(len(probmap.ravel())),
                        probmap.shape,
                    )
                )
            else:
                self._gaze_loc = np.array(
                    np.unravel_index(
                        np.random.choice(
                            len(probmap.ravel()), p=probmap.ravel() / probmapsum
                        ),
                        probmap.shape,
                    )
                )
        else:
            # otherwise, we do fixational eye movements
            gaze_drift = np.array(
                [
                    int(np.round(np.random.normal(0, self.params["drift_sig"]))),
                    int(np.round(np.random.normal(0, self.params["drift_sig"]))),
                ]
            )
            # ...and smooth pursuit! (if not background and moves)
            # ASSUMPTION: only one obj can be foveated - otherwise must take the average!
            for obj in self.video_data["object_list"][1:]:
                if obj.foveated:
                    gaze_drift += obj.shift[self._current_frame]
            self._gaze_loc += gaze_drift

        # make sure that gaze is always on video
        self._gaze_loc[0] = max(min(self._gaze_loc[0], self.Dataset.VID_SIZE_Y - 1), 0)
        self._gaze_loc[1] = max(min(self._gaze_loc[1], self.Dataset.VID_SIZE_X - 1), 0)

    def write_sgl_output_gif(self, storagename, slowgif=False, dpi=100):
        """
        Visualize the single trial run and save it as a gif.

        Illustrates what is happening in the 5 modules for each frame.

        :param storagename: Name of the file, will be appended to outputpath +".gif"
        :type storagename: str
        :param slowgif: If true, store it with 10fps, otherwise 30fps, defaults to False
        :type slowgif: bool, optional
        :param dpi: DPI for each frame when created and stored gif, defaults to 100
        :type dpi: int, optional
        """
        assert (
            self.params["sglrun_return"] == True
        ), "`writeOutputVis` is only available for single trial runs!"
        if hasattr(self.Dataset, "outputpath"):
            outputpath = self.Dataset.outputpath + storagename
        else:
            outputpath = f"{self.Dataset.PATH}results/{storagename}"

        vidname = next(iter(self.result_dict))
        runname = next(iter(self.result_dict[vidname]))

        vidlist = self.Dataset.load_videoframes(vidname)
        F_max = np.max(self.video_data["feature_maps"])

        @gif.frame
        def frame(f):
            fig, axs = plt.subplots(2, 3, figsize=(10, 4.5), dpi=dpi)
            axs[0, 0].imshow(
                self.video_data["feature_maps"][f], cmap="inferno", vmin=0
            )  # , vmax=F_max)
            axs[0, 0].set_title("(I) Scene features")
            axs[0, 1].imshow(self._all_sens[f], cmap="bone", vmin=0, vmax=1)
            axs[0, 1].set_title("(II) Visual sensitivity")
            axs[1, 0].imshow(vidlist[f + 1], alpha=0.5)
            axs[1, 0].set_title(f"Video frame {f:03d}")

            cmaps = ["Blues", "Oranges", "Reds", "Greens", "Purples"]
            IOR_map = np.ones_like(self.video_data["feature_maps"][f])
            evidence_map = self.video_data["feature_maps"][f] * self._all_sens[f]
            for o in range(len(self.video_data["object_list"])):
                temp = self.video_data["object_list"][o].object_maps[f].copy() * 1.0
                IOR_map = IOR_map - temp * self._all_iors[f][o]
                temp[temp == 0] = np.nan
                axs[1, 0].imshow(temp, cmap=cmaps[(o) % 5], vmin=0, vmax=1, alpha=0.5)
                axs[1, 1].imshow(
                    temp * self._all_dvs[f][o],
                    cmap=cmaps[(o) % 5],
                    vmin=0,
                    vmax=self.params["ddm_thres"],
                )
                axs[1, 2].imshow(
                    temp * evidence_map,  # * self._all_iors[f][o]
                    cmap=cmaps[(o) % 5],
                    vmin=0,
                    # vmax=self.params["ddm_thres"],
                )

            axs[0, 2].imshow(IOR_map, cmap="bone", vmin=0, vmax=1)
            axs[0, 2].set_title("(III) Scanpath history")
            axs[1, 1].set_title("(IV) Decision making")
            axs[1, 2].set_title("(V) Gaze update")
            for ax in axs.flat:
                ax.scatter(
                    self.result_dict[vidname][runname]["gaze"][f][1],
                    self.result_dict[vidname][runname]["gaze"][f][0],
                    s=300,
                    c="green",
                    marker="x",
                    lw=2,
                )
                ax.axis("off")
            axs[1, 2].scatter(
                self.result_dict[vidname][runname]["gaze"][f + 1][1],
                self.result_dict[vidname][runname]["gaze"][f + 1][0],
                s=300,
                facecolors="none",
                edgecolors="r",
                marker="o",
                lw=3,
                alpha=0.75,
            )

            axs[1, 2].arrow(
                self.result_dict[vidname][runname]["gaze"][f][1],
                self.result_dict[vidname][runname]["gaze"][f][0],
                self.result_dict[vidname][runname]["gaze"][f + 1][1]
                - self.result_dict[vidname][runname]["gaze"][f][1],
                self.result_dict[vidname][runname]["gaze"][f + 1][0]
                - self.result_dict[vidname][runname]["gaze"][f][0],
                head_width=15,
                head_length=15,
                fc="k",
                ec="k",
                lw=2,
                alpha=0.75,
                length_includes_head=True,
            )
            fig.set_facecolor("lightgrey")
            plt.tight_layout()

        out = [frame(i) for i in range(len(self._all_dvs))]
        if slowgif:
            gif.save(out, outputpath + "_slow.gif", duration=100)
            # gif.save(out, f"videos/objvideo_{name}_{vidname}_thres_dv{thres_dv}_sig_dv{sig_dv}_ior_decay{ior_decay}_att_obj{att_obj}_att_dva{att_dva}_rs{rs}_slow.gif", duration=100)
            print(f"Saved to {outputpath}_slow.gif")
        else:
            gif.save(out, outputpath + ".gif", duration=33)
            print(f"Saved to {outputpath}.gif")
            # gif.save(out, f"videos/objvideo_{name}_{vidname}_thres_dv{thres_dv}_sig_dv{sig_dv}_ior_decay{ior_decay}_att_obj{att_obj}_att_dva{att_dva}_rs{rs}.gif", duration=33)
