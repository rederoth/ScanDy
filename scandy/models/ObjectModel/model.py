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
        self.video_data = None
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
        assert self.video_data is not None, "Video data not loaded"
        for obj in self.video_data["object_list"]:
            obj.set_initial_state()
        self._scanpath = []
        self._f_sac = []
        self._current_frame = 0
        self._gaze_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._prev_loc = np.zeros(2, int)  # not used, but potentially for IOR
        self._new_target = None
        self._feature_map.fill(0.0)
        self._sens_map.fill(0.0)
        self._ior_objs = np.zeros(len(self.video_data["object_list"]))
        self._decision_objs = np.zeros(len(self.video_data["object_list"]))
        # Stored maps for a single run for visualization
        self._all_dvs = []
        self._all_iors = []
        self._all_sens = []

    def update_features(self):
        """
        Current frame of the features, loaded as specified in the model parameters.

        TRYOUT: could use a time-dependent center bias
        """
        assert self.video_data is not None, "Video data not loaded"
        self._feature_map = self.video_data["feature_maps"][self._current_frame]

    def update_sensitivity(self):
        """
        Visual sensitivity map, Gaussian spread around the gaze point, with the
        currently foveated object (if not background) having sensitivity 1.

        TRYOUT: increase sensitivity for objects that are close to the decision
        threshold (i.e. presaccadic attention)
        """
        assert self.params is not None, "Model parameters not loaded"
        assert self.video_data is not None, "Video data not loaded"

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
        Object-based inhibition of return, updated for all objects in the scene.

        TRYOUT: make location-based map, as proposed by Krüger et al. (2013)
        """
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
        assert self.video_data is not None, "Video data not loaded"
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



    #########
    ## TODO !!!
    #########

    def write_sgl_output_gif(self, storagename, slowgif=False):
        """TODO: Write nice and general visualization function based on self.visual_vars
        This function produces a GIF for a single run (seed) for a single video.
        It shows the evidence, sensitivity, feature, IOR, and object/OF map.
        """
        assert (
            self.params["sglrun_return"] == True
        ), "`writeOutputVis` is only available for single trial runs!"
        if hasattr(self.Dataset, "outputpath"):
            outputpath = self.Dataset.outputpath + storagename
        else:
            hasattr(self.Dataset, "PATH"), "Dataset has no defined PATH"
            outputpath = f"{self.Dataset.PATH}results/{storagename}"

        # video_data not necessary, can be loaded in function based on knowledge of video name (key in result_dict)
        vidname = next(iter(self.result_dict))
        runname = next(iter(self.result_dict[vidname]))
        res_dict = self.result_dict[vidname][runname]

        video_data = self.get_videodata(vidname)
        vidlist = self.Dataset.get_videoframes(vidname)
        feature_maps = video_data["feature_maps"]
        F_max = np.max(feature_maps)
        # TODO: do 0th element in simulation code, not here...
        DVs = res_dict["DVs"]
        IORs = res_dict["IORs"]

        @gif.frame
        def frame(f):
            S = uf.gaussian_2d(
                res_dict["gaze"][f][1],
                res_dict["gaze"][f][0],
                video_data["VID_SIZE_X"],
                video_data["VID_SIZE_Y"],
                self.params["att_dva"] * video_data["DVA_TO_PX"],
                sumnorm=False,
            )
            F = feature_maps[f]
            # get the foveated object - without radius, already implemented
            # (since that is how it is calculated, radius is only for analysis)
            fovobj = uf.object_at_position(
                video_data["segmentation_masks"][f],
                res_dict["gaze"][f][1],
                res_dict["gaze"][f][0],
                radius=None,
            )
            # sensitivity on 1 across the foveated object mask (unless it's the Ground)
            if fovobj != "Ground":
                # fovobj is a string "Object X" --> extract X
                fov_obj_id = int(fovobj[-1])
                S[video_data["object_list"][fov_obj_id].object_maps[f]] = 1
            else:
                fov_obj_id = 0

            fig = plt.figure(figsize=(7, 8))
            gs = fig.add_gridspec(4, 2)
            ax0 = fig.add_subplot(gs[0:2, :])
            ax1 = fig.add_subplot(gs[2, 0])
            ax2 = fig.add_subplot(gs[2, 1])
            ax3 = fig.add_subplot(gs[3, 0])
            ax4 = fig.add_subplot(gs[3, 1])
            ax0.imshow(vidlist[f])
            # TODO: Make it work for >5 objects....
            cmaps = ["Blues", "Oranges", "Reds", "Greens", "Purples"]
            IOR_map = np.ones_like(F)
            for o in range(DVs.shape[1]):
                temp = video_data["object_list"][o].object_maps[f].copy() * 1.0
                # temp = res_dict['DVs'][o][f]
                if o == fov_obj_id:
                    if o > 0:
                        IOR_map = IOR_map - temp * IORs[f, o] * self.params["ior_inobj"]
                else:
                    IOR_map = IOR_map - temp * IORs[f, o]
                temp[temp == 0] = np.nan
                ax4.imshow(
                    temp,
                    cmap=cmaps[(o) % 5],
                    vmin=0,
                    vmax=1,  # alpha=0.5
                )  # vmax = ddmpar['thres_dv']
                ax0.imshow(
                    temp * DVs[f, o],
                    cmap=cmaps[(o) % 5],
                    vmin=0,
                    vmax=self.params["ddm_thres"],
                    alpha=0.5,
                )

            # ax0.imshow(DVs[f], vmin=0, vmax=thres_dv, cmap="Reds", alpha=0.5)
            ax0.set_title("Decision Varaible")
            ax1.imshow(S, cmap="bone", vmin=0, vmax=1)
            ax1.set_title("Sensitivity")
            ax2.imshow(F, cmap="inferno", vmin=0, vmax=F_max)  # , alpha=0.5)
            ax2.set_title("Features (incl. center bias)")
            ax3.imshow(IOR_map, cmap="bone", vmin=0, vmax=1)
            ax3.set_title("Inhibition of Return")
            # ax4.imshow(masks)
            ax4.set_title("Object Masks")
            for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4]):
                if i == 0:
                    markersize = 800
                else:
                    markersize = 200
                ax.scatter(
                    res_dict["gaze"][f][1],
                    res_dict["gaze"][f][0],
                    s=markersize,
                    c="green",
                    marker="+",
                )
                ax.axis("off")
            plt.tight_layout()
            # plt.savefig(f'videos/vid_fig/{f:03d}.png', pad_inches=0, dpi=150); plt.close()

        nframes = feature_maps.shape[0]
        out = [frame(i) for i in range(nframes)]
        if slowgif:
            gif.save(out, outputpath + "_slow.gif", duration=100)
            # gif.save(out, f"videos/objvideo_{name}_{vidname}_thres_dv{thres_dv}_sig_dv{sig_dv}_ior_decay{ior_decay}_att_obj{att_obj}_att_dva{att_dva}_rs{rs}_slow.gif", duration=100)
            print(f"Saved to {outputpath}_slow.gif")
        else:
            gif.save(out, outputpath + ".gif", duration=33)
            print(f"Saved to {outputpath}.gif")
            # gif.save(out, f"videos/objvideo_{name}_{vidname}_thres_dv{thres_dv}_sig_dv{sig_dv}_ior_decay{ior_decay}_att_obj{att_obj}_att_dva{att_dva}_rs{rs}.gif", duration=33)
