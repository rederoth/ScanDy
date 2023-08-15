# from . import parameterDict as par
import os
import numpy as np
import matplotlib.pyplot as plt
import gif
import flow_vis

# from . import videoRun as vr
from ..model import Model
from ...utils import functions as uf
from neurolib.utils.collections import dotdict


class LocationModel(Model):
    """
    Basic location-based model: potential saccade targets are pixels.
    """

    name = "loc"
    description = "Basic location-based model with calculations on a pixel level."

    def __init__(self, Dataset, params=None, preload_res_df=None):
        """
        Initialize the model. IOR and decision variables for each pixel.

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

        # pixel-based maps for the different moduls
        self._feature_map = np.zeros((self.Dataset.VID_SIZE_Y, self.Dataset.VID_SIZE_X))
        self._ior_map = np.zeros_like(self._feature_map)
        self._sens_map = np.zeros_like(self._feature_map)
        self._decision_map = np.zeros_like(self._feature_map)
        # Stored maps for a single run for visualization
        self._all_dvs = []
        self._all_iors = []
        self._all_sens = []
        # self._all_features = [] # only if features are modified, otherwise save memory

    def load_default_modelparams(self):
        """
        Defines the model specifications, most importantly the feature map (defaults
        to low-level features with anisotropic center bias).
        Free parameters that should be adapted are ior_decay, ior_dva, att_dva,
        ddm_thres, ddm_sig.

        :return: Parameter dictionary
        :rtype: dict
        """
        par = dotdict({})

        # random seed for a scanpath simulation, is set in the model.run function
        par.rs = None
        # starting position, default is center
        par.startpos = np.array(
            [self.Dataset.VID_SIZE_X // 2, self.Dataset.VID_SIZE_Y // 2]
        )
        par.ior_decay = 1.0 * self.Dataset.FPS
        # Spread of Gaussian inhibition around previous fixation positions
        par.ior_dva = 2.0
        # Spread of Gaussian visual sensitivity around the gaze point
        par.att_dva = 6.0
        # Drift-diffusion model
        par.ddm_thres = 0.3
        par.ddm_sig = 0.01
        # par.ddm_reset = 0  # add this param if update_decision is changed!
        # Oculomotor drift
        par.drift_sig = self.Dataset.DVA_TO_PX / 8.0
        # Since optical flow somewhat carries object info, we might not want to use it here
        par.use_flow = True
        # The default LocationModel uses no objects, but we still allow it to
        # load objects here to make it easier to modify
        par.use_objects = False
        par.use_objectfiles = False
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
        self._scanpath = []
        self._f_sac = []
        self._current_frame = 0
        self._new_target = None
        self._prev_gaze_loc = np.zeros(2, int)
        self._gaze_loc = np.zeros(2, int)
        self._decision_map.fill(0.0)
        self._ior_map.fill(0.0)
        self._sens_map.fill(0.0)
        self._feature_map.fill(0.0)
        # Stored maps for a single run for visualization
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
        Visual sensitivity map, Gaussian spread around the gaze point.

        TRYOUT: check if object is foveated and if so, spread across mask
        """
        assert self.params is not None, "Model parameters not loaded"
        gaze_gaussian = uf.gaussian_2d(
            self._gaze_loc[0],
            self._gaze_loc[1],
            self.Dataset.VID_SIZE_X,
            self.Dataset.VID_SIZE_Y,
            self.params["att_dva"] * self.Dataset.DVA_TO_PX,
        )
        self._sens_map = gaze_gaussian
        if self.params["sglrun_return"]:
            self._all_sens.append(self._sens_map.copy())

    def update_ior(self):
        """
        Modul (III): Scanpath history
        Location-based inhibition of return.

        TRYOUT: instead of setting the inhibition from zero to one, increase it
        over time.
        """
        assert self.params is not None, "Model parameters not loaded"
        # if saccade was done in the last frame, add new inhibition
        if self._new_target is not None:
            inhibition = uf.gaussian_2d(
                self._prev_gaze_loc[0],
                self._prev_gaze_loc[1],
                self.Dataset.VID_SIZE_X,
                self.Dataset.VID_SIZE_Y,
                self.params["ior_dva"] * self.Dataset.DVA_TO_PX,
            )
            # TRYOUT: instead of fully inhibiting, scale with self.params["ior_magnitude"]
            self._ior_map = np.clip(self._ior_map + inhibition, 0, 1)
        # in every timestep, decrease the inhibition linearly with dI = - r dt
        self._ior_map = np.clip(self._ior_map - 1.0 / self.params["ior_decay"], 0, 1)
        if self.params["sglrun_return"]:
            self._all_iors.append(self._ior_map.copy())

    def update_decision(self):
        """
        Modul (IV): Decision making
        Updates the pixel-wise accumulated evidence V, based on the feature map F,
        the sensitivity S, and the inhibition I of the current frame.

        TRYOUT: After a decision is made, instead of reseting the decision variables
        to zero, keep some evidence in memory and multiply with self.params["ddm_reset"].
        """
        assert self.params is not None, "Model parameters not loaded"

        if self._cur_fov_frac > 0.0:
            # update decision variables
            px_evidence = self._sens_map * self._feature_map * (
                1 - self._ior_map
            ) + np.random.normal(0, self.params["ddm_sig"], self._decision_map.shape)
            self._decision_map = self._decision_map + (px_evidence * self._cur_fov_frac)

            max_dv = np.max(self._decision_map)
            if max_dv > self.params["ddm_thres"]:
                # find the pixel that crossed the threshold ==> becomes the target
                choice_idx = np.unravel_index(
                    np.argmax(self._decision_map), self._decision_map.shape
                )
                self._new_target = np.array([choice_idx[1], choice_idx[0]], dtype=int)

                # fraction of dt after which the threshold would be crossed
                dt_frac_sac = (max_dv - self.params["ddm_thres"]) / px_evidence[
                    self._new_target[1], self._new_target[0]
                ]
                self._cur_waiting_time = np.clip(dt_frac_sac, 0, 1) * self._dt

                # reset decision variables for all pixels
                self._decision_map.fill(0)
                # self._decision_map *= self.params["ddm_reset"]
            else:
                self._new_target = None
        else:
            self._new_target = None

        if self.params["sglrun_return"]:
            self._all_dvs.append(self._decision_map)

    def update_gaze(self):
        """
        Module(V): Gaze update
        Either make saccade to the target pixel, or follow the optical flow while
        doing smooth pursuit and fixational eye movements.

        TRYOUT: Saccade could be inaccurate, might lead to follow-up saccades.
        """
        assert self.params is not None, "Model parameters not loaded"
        self._prev_gaze_loc = self._gaze_loc.copy()  # for IOR
        if self._new_target is not None:
            # if there is a new saccade target, set gaze location accurately to target
            self._gaze_loc = self._new_target.copy()
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
                        self._current_frame, self._gaze_loc[1], self._gaze_loc[0]
                    ],
                    int,
                )
            self._gaze_loc += np.array([gaze_drift[0], gaze_drift[1]])

        # make sure that gaze is always on video
        self._gaze_loc[0] = max(min(self._gaze_loc[0], self.Dataset.VID_SIZE_X - 1), 0)
        self._gaze_loc[1] = max(min(self._gaze_loc[1], self.Dataset.VID_SIZE_Y - 1), 0)

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
                self.video_data["feature_maps"][f], cmap="inferno", vmin=0, vmax=F_max
            )
            axs[0, 0].set_title("(I) Scene features")
            axs[0, 1].imshow(self._all_sens[f], cmap="bone", vmin=0, vmax=1)
            axs[0, 1].set_title("(II) Visual sensitivity")
            axs[0, 2].imshow(1 - self._all_iors[f], cmap="bone", vmin=0, vmax=1)
            axs[0, 2].set_title("(III) Scanpath history")
            axs[1, 0].imshow(vidlist[f + 1])
            axs[1, 0].set_title(f"Video frame {f:03d}")
            axs[1, 1].imshow(
                self._all_dvs[f],
                vmin=0,
                vmax=self.params["ddm_thres"],
                cmap="Reds",
                alpha=0.5,
            )
            axs[1, 1].set_title("(IV) Decision making")
            flow_color = flow_vis.flow_to_color(
                self.video_data["flow_maps"][f], convert_to_bgr=False
            )
            axs[1, 2].imshow(flow_color)
            axs[1, 2].set_title("(V) Gaze update")
            for ax in axs.flat:
                ax.scatter(
                    self.result_dict[vidname][runname]["gaze"][f][0],
                    self.result_dict[vidname][runname]["gaze"][f][1],
                    s=300,
                    c="green",
                    marker="x",
                    lw=2,
                )
                ax.axis("off")
            axs[1, 2].scatter(
                self.result_dict[vidname][runname]["gaze"][f + 1][0],
                self.result_dict[vidname][runname]["gaze"][f + 1][1],
                s=300,
                facecolors="none",
                edgecolors="r",
                marker="o",
                lw=3,
                alpha=0.75,
            )
            axs[1, 2].arrow(
                self.result_dict[vidname][runname]["gaze"][f][0],
                self.result_dict[vidname][runname]["gaze"][f][1],
                self.result_dict[vidname][runname]["gaze"][f + 1][0]
                - self.result_dict[vidname][runname]["gaze"][f][0],
                self.result_dict[vidname][runname]["gaze"][f + 1][1]
                - self.result_dict[vidname][runname]["gaze"][f][1],
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
