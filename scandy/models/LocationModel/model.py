# from . import parameterDict as par
import os
import numpy as np
import matplotlib.pyplot as plt
import gif
import flow_vis as fv

# from . import videoRun as vr
from ..model import Model
from ...utils import functions as uf
from neurolib.utils.collections import dotdict

# from ...utils import visualizations as uv


class LocationModel(Model):
    """
    Basic location-based model that accumulates evidence for every pixel.
    """

    name = "loc"
    description = "Basic location-based model that accumulates evidence for every pixel"

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
        self._decision_map = np.zeros(
            (self.Dataset["VID_SIZE_Y"], self.Dataset["VID_SIZE_X"])
        )
        self._ior_map = np.zeros_like(self._decision_map)
        self._sens_map = np.zeros_like(self._decision_map)
        self._feature_map = np.zeros_like(self._decision_map)
        self._scanpath = []
        self._f_sac = []
        self._gaze_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._new_target = None
        self._prev_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._current_frame = 0
        # Stored maps for a single run for visualization
        self._all_dvs = []
        self._all_iors = []
        self._all_sens = []
        # self._all_features = [] # only if features are modified, otherwise save memory

    def load_default_modelparams(self):
        """Load parameter dictionary for running the PixelDDM.
        :params Dataset: Used dataset, important attributes are FPS and DVA_TO_PX.
        :returns params_dict: Default model parameters as dict.
        """
        par = dotdict({})

        # random seed for a scanpath simulation, is set in the model.run function
        par.rs = None
        # starting position, default is center
        par.startpos = np.array(
            [self.Dataset.VID_SIZE_Y // 2, self.Dataset.VID_SIZE_X // 2]
        )
        par.ior_decay = 1.0 * self.Dataset.FPS
        # Spread of Gaussian inhibition around previous fixation positions
        par.ior_dva = 2.0
        # Spread of Gaussian visual sensitivity around the gaze point
        par.att_dva = 6.0
        # Drift-diffusion model
        par.ddm_thres = 0.3
        par.ddm_sig = 0.01
        par.ddm_reset = 0
        # Oculomotor drift
        par.drift_sig = self.Dataset.DVA_TO_PX / 8.0
        # Since optical flow somewhat carries object info, we might not want to use it here
        par.use_flow = True
        # The default LocationModel uses no objects, but we still allow it to
        # load objects here to make it easier to modify
        par.use_objects = False
        # Choose the way how IOR should be done TODO: implement alternatives
        par.ior_method = "targetLocConstDecay"
        # Choose the used feature map and whether a center bias should be used
        par.centerbias = "anisotropic_default"
        par.featuretype = "molin"
        assert os.path.isdir(
            f"{self.Dataset.featuremaps}{par.featuretype}/"
        ), f"No stored features at {self.Dataset.featuremaps}{par.featuretype}/"
        # Determines how verbose the return dictionary is, detailed info only for single runs
        par.sglrun_return = True

        # params_dict = par.__dict__
        # self.params = par
        return par

    # def load_videodata(self, videoname):
    #     """
    #     TODO: move to base class?!?!

    #     Provided the model parameters, load everything that is necessary to run
    #     the model for a given video.

    #     :param videoname: Single video for which the model should be run
    #     :type videoname: str
    #     :return: Parameter dictionary with all data neccessary to run the model for one video
    #     :rtype: dict
    #     """
    #     viddata = dotdict({})
    #     assert self.params is not None, "Model parameters not loaded"

    #     viddata.videoname = videoname
    #     # fetch info on featuretype and cb to load the feature map
    #     centerbias = self.params["centerbias"]
    #     featuretype = self.params["featuretype"]
    #     viddata.feature_maps = self.Dataset.load_featuremaps(
    #         videoname, featuretype, centerbias
    #     )
    #     if self.params["use_flow"]:
    #         viddata.flow_maps = self.Dataset.load_flowmaps(videoname)
    #     if self.params["use_objects"]:
    #         viddata.object_masks = self.Dataset.load_objectmasks(videoname)
    #     viddata.nframes = self.Dataset.video_frames[videoname]
    #     self.video_data = viddata  # save to class
    #     # return viddata

    def update_features(self):
        # TRYOUT: could use a time-dependent center bias
        assert self.video_data is not None, "Video data not loaded"
        self._feature_map = self.video_data["feature_maps"][self._current_frame]

    def update_sensitivity(self):
        # TRYOUT: check if object is foveated and if so, spread across mask
        assert self.params is not None, "Model parameters not loaded"
        gaze_gaussian = uf.gaussian_2d(
            self._gaze_loc[1],
            self._gaze_loc[0],
            self.Dataset["VID_SIZE_X"],
            self.Dataset["VID_SIZE_Y"],
            self.params["att_dva"] * self.Dataset["DVA_TO_PX"],
            sumnorm=False,
        )
        self._sens_map = gaze_gaussian
        # return gaze_gaussian

    # def update_ior(self, ior_map, gaze_loc, new_target, prev_loc):
    def update_ior(self):
        assert self.params is not None, "Model parameters not loaded"
        # if saccade was done in the last frame, add new inhibition
        if self._new_target is not None:
            inhibition = uf.gaussian_2d(
                self._prev_loc[1],
                self._prev_loc[0],
                self.Dataset["VID_SIZE_X"],
                self.Dataset["VID_SIZE_Y"],
                self.params["ior_dva"] * self.Dataset["DVA_TO_PX"],
            )
            # TRYOUT: instead of fully inhibiting, scale with self.params["ior_magnitude"]
            self._ior_map = np.clip(self._ior_map + inhibition, 0, 1)
        # in every timestep, decrease the inhibition linearly with dI = - r dt
        self._ior_map = np.clip(self._ior_map - 1.0 / self.params["ior_decay"], 0, 1)
        # return self._ior_map

    # def update_decision(self, V, F, S, I):
    def update_decision(self):
        """
        Modul (IV): Decision making

        Updates the pixel-wise accumulated evidence V, based on the feature map F,
        the sensitivity S, and the inhibition I of the current frame.

        :param V: Pixel-wise accumulated evidence,
        :type V: np.ndarray
        :param F: Feature map of the current frame, loaded in modul I
        :type F: np.ndarray
        :param S: Sensitivity map of the current frame, updated in modul II
        :type S: np.ndarray
        :param I: Inhibition map of the current frame, updated in modul III
        :type I: np.ndarray
        """
        assert self.params is not None, "Model parameters not loaded"

        px_evidence = self._sens_map * self._feature_map * (1 - self._ior_map)
        self._decision_map += px_evidence + np.random.normal(
            0, self.params["ddm_sig"], self._decision_map.shape
        )

        if np.max(self._decision_map) > self.params["ddm_thres"]:
            # find the pixel that crossed the threshold ==> becomes the target
            self._new_target = np.array(
                np.unravel_index(
                    np.argmax(self._decision_map), self._decision_map.shape
                ),
                int,
            )
            # reset decision variables for all pixels
            # TRYOUT: We could set to self.params["ddm_reset"] instead of 0
            self._decision_map *= 0
        else:
            self._new_target = None
        # return V, target

    # def update_gaze(self, frame, gaze_loc, target):
    def update_gaze(self):
        assert self.params is not None, "Model parameters not loaded"
        assert self.video_data is not None, "Video data not loaded"
        self._prev_loc = self._gaze_loc.copy()
        if self._new_target is not None:
            # if there is a new saccade target, set gaze location accurately to target
            self._gaze_loc = self._new_target
        else:
            # otherwise, we do fixational eye movements
            gaze_drift = np.array(
                [
                    int(np.round(np.random.normal(0, self.params["drift_sig"]))),
                    int(np.round(np.random.normal(0, self.params["drift_sig"]))),
                ]
            )
            # ...and smooth pursuit, but only if self.params["use_flow"]==True
            # flow maps has structure: [frames-1, ydim, xdim, 2=[xmap, ymap]]
            if self.params["use_flow"]:
                gaze_drift += np.array(
                    self.video_data["flow_maps"][
                        self._current_frame, self._gaze_loc[0], self._gaze_loc[1]
                    ],
                    int,
                )
            self._gaze_loc += np.array([gaze_drift[1], gaze_drift[0]])

        # make sure that gaze is always on video
        self._gaze_loc[0] = max(
            min(self._gaze_loc[0], self.Dataset["VID_SIZE_Y"] - 1), 0
        )
        self._gaze_loc[1] = max(
            min(self._gaze_loc[1], self.Dataset["VID_SIZE_X"] - 1), 0
        )

    def reinit_for_sgl_run(self):
        """
        Reinitialize the model such that all parameters which might be updated
        in a simulation run are back in their initial state.
        """
        self._decision_map *= 0
        self._ior_map *= 0
        self._sens_map *= 0
        self._feature_map *= 0
        self._scanpath = []
        self._f_sac = []
        self._current_frame = 0
        self._new_target = None
        self._prev_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._gaze_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._all_dvs = []
        self._all_iors = []
        self._all_sens = []


    def sgl_vid_run(self, videoname, force_reload=False, visualize=False):

        assert self.params is not None, "Model parameters not loaded"
        # load the relevant data for videoname if not already loaded (or forced)
        if self.video_data is None or force_reload:
            self.load_videodata(videoname)
            print("Loaded video (None or reload) for", videoname)
        elif self.video_data.videoname != videoname:
            self.load_videodata(videoname)
            print("Loaded video (new name) for", videoname)
        assert self.video_data is not None, "Video data not loaded"

        # If provided, set random seed.
        if self.params.rs:
            np.random.seed(self.params.rs)

        # all_dvs = []  # np.zeros_like(self.video_data.feature_maps)
        # all_iors = []  # np.zeros_like(self.video_data.feature_maps)

        # reinit all variables
        self.reinit_for_sgl_run()

        # set initial gaze location
        self._gaze_loc = self.params["startpos"].copy()
        self._scanpath.append(self._gaze_loc.copy())

        # Loop through all frames and run all modules
        # no new location in prediction in last frame => len(scanpath)=nframes
        for f in range(self.video_data["nframes"] - 1):

            self._current_frame = f
            # feature_map = self.video_data["feature_maps"][f]
            self.update_features()
            self.update_sensitivity()  # f, gaze_loc)
            self.update_ior()  # ior_map, gaze_loc, new_target, prev_loc)
            # store maps for analysis & visu, but only if its a single run!
            if self.params["sglrun_return"]:
                self._all_dvs.append(self._decision_map)
                self._all_iors.append(self._ior_map)
                self._all_sens.append(self._sens_map)
            # update decision map based on modules (I-III)
            self.update_decision()  # dv_map, feature_map, sensitivity_map, ior_map)
            self.update_gaze()  # f, gaze_loc, new_target)

            # store when a saccade was made in list
            if self._new_target is not None:
                self._f_sac.append(f)

            # add updated gaze location to scanpath
            self._scanpath.append(self._gaze_loc.copy())

            if visualize:
                pass

        res_dict = {
            "gaze": np.asarray(self._scanpath),
            "f_sac": np.asarray(self._f_sac),
        }
        return res_dict

    #########
    ## TODO !!!
    ## Would be better to pass the sensitivity map...
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
            outputpath = f"{self.Dataset.PATH}results/{storagename}"

        # video_data not necessary, can be loaded in function based on knowledge of video name (key in result_dict)
        vidname = next(iter(self.result_dict))
        runname = next(iter(self.result_dict[vidname]))
        res_dict = self.result_dict[vidname][runname]
        thres_dv = self.params["ddm_thres"]
        # sig_dv = self.params['ddm_sig']
        # ior_decay = self.params['ior_decay']
        # att_obj = self.params['intraobjatt']
        # att_dva = self.params['sensitivity']
        # rs = ddmpar['rs']

        video_data = self.load_videodata(vidname)
        vidlist = self.Dataset.load_videoframes(vidname)
        feature_maps = self.video_data["feature_maps"]
        F_max = np.max(feature_maps)
        # TODO: do 0th element in simulation code, not here...
        DVs = np.concatenate(
            (
                np.zeros((1, self.Dataset["VID_SIZE_Y"], self.Dataset["VID_SIZE_X"])),
                res_dict["DVs"],
            )
        )
        IORs = np.concatenate(
            (
                np.ones((1, video_data["VID_SIZE_Y"], video_data["VID_SIZE_X"])),
                res_dict["IORs"],
            )
        )
        OFs = np.concatenate(
            (
                np.zeros((1, video_data["VID_SIZE_Y"], video_data["VID_SIZE_X"], 2)),
                video_data["flow_maps"],
            )
        )

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

            fig = plt.figure(figsize=(7, 8), dpi=150)
            gs = fig.add_gridspec(4, 2)
            ax0 = fig.add_subplot(gs[0:2, :])
            ax1 = fig.add_subplot(gs[2, 0])
            ax2 = fig.add_subplot(gs[2, 1])
            ax3 = fig.add_subplot(gs[3, 0])
            ax4 = fig.add_subplot(gs[3, 1])
            ax0.imshow(vidlist[f])
            ax0.imshow(DVs[f], vmin=0, vmax=thres_dv, cmap="Reds", alpha=0.5)
            ax0.set_title("Decision Varaible")
            ax1.imshow(S, cmap="bone", vmin=0, vmax=1)
            ax1.set_title("Sensitivity")
            ax2.imshow(F, cmap="inferno", vmin=0, vmax=F_max)  # , alpha=0.5)
            ax2.set_title("Features (incl. center bias)")
            ax3.imshow(IORs[f], cmap="bone", vmin=0, vmax=1)
            ax3.set_title("Inhibition of Return")
            flow_color = fv.flow_to_color(OFs[f], convert_to_bgr=False)
            ax4.imshow(flow_color)
            ax4.set_title("Optical Flow")
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
                    marker="+",  # type: ignore
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
