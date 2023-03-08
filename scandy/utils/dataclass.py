import os
import numpy as np
import pandas as pd
import imageio
import yaml

from .functions import anisotropic_centerbias


class Dataset:
    """
    This class loads and handles the human scanpath data (gt_foveation_df), the
    videos of the dataset and all precomputed maps, including object segmentations,
    optical flow, and different saliency meassures.
    Attributes contain important information about the data, like the size of
    the videos (#frames, size in x & y) and the conversion from pixels to degrees
    visual angle (dva).

    Required information in the dataset:
        * PATH: Path to the data where all if stored in predefined schema (see below)
        * FPS: Frames per second of the videos
        * PX_TO_DVA: Conversion from pixels to degrees visual angle
        * gt_foveation_df: Dataframe with the ground truth scanpaths

    The following information & paths are derived or set if not explicitely given(-> indicates how it is inferred):
        * videos: -> {PATH}videos/, original videos (only used for visualization)
        * featuremaps: -> {PATH}featuremaps/, maps for features (have subfolders for different feature extraction methods, like 'molin')
        * objectmasks: -> {PATH}polished_segmentation/, Object segmentation masks (polished, px values are time consistent object ids)
        * flowmaps: -> {PATH}optical_flow/, Optical flow maps
        * nssmaps: -> {PATH}gt_fov_maps_333/, Ground truth fixation maps, normalized for calculating NSS
        * used_videos: -> names in objectmasks, list of videos used for modeling, usually limited by segmentations
        * FRAMES_ALL_VIDS: -> objectmasks[used_videos].shape[0], Number of frames either given for all (int) or inferred for each (list)
        * VID_SIZE_Y, VID_SIZE_X: -> objectmasks[0].shape[1:] are assumed to be the same for all videos (for convenience).
        * video_frames: -> FRAMES_ALL_VIDS, Number of frames for each video (dict)
        * RADIUS_OBJ_GAZE: Tolerance radius of the object around the gaze point (in dva), default: 1.0
        * trainset: List of videos used for training, default: used_videos
        * testset: List of videos used for testing, default: None
        * gt_fovframes_nss_df: Dataframe with NSS values of the ground truth scanpaths, only needed if NSS is of interest
    """

    def __init__(self, dataconfig):
        """
        Load the important information from the provided dataset.
        :param dataconfig: Dictionary that contains the most important info about the dataset.
        :type dataconfig: dict or str (path to yaml file)
        """
        # load the dataconfig, either as dictionary or from a yaml file
        if isinstance(dataconfig, dict):
            datadict = dataconfig
        elif isinstance(dataconfig, str):
            assert dataconfig.endswith(".yaml") or dataconfig.endswith(
                ".yml"
            ), "Path is not a YAML file!"
            datadict = self.load_yaml(dataconfig)
            assert isinstance(datadict, dict), "YAML file was not converted to dict!"
        else:
            raise TypeError(
                "Input must be a dictionary or a string path to a YAML file"
            )

        # Inputs that cannot easily be derived from maps must be given:
        assert "FPS" in datadict, f"FPS has to be provided as key in dataconfig!"
        self.FPS = datadict["FPS"]
        assert (
            "PX_TO_DVA" in datadict
        ), f"PX_TO_DVA has to be provided as key in dataconfig!"
        self.PX_TO_DVA = datadict["PX_TO_DVA"]
        self.DVA_TO_PX = 1.0 / self.PX_TO_DVA

        # Path to the data where all if stored in predefined schema (see below)
        assert (
            "PATH" in datadict
        ), f"PATH to data has to be provided as key in dataconfig!"
        self.PATH = datadict["PATH"]
        # derive other paths if not explicitely given...
        # maps for features (have subfolders for different feature extraction methods, like 'molin')
        if "featuremaps" in datadict:
            self.featuremaps = datadict["featuremaps"]
        else:
            self.featuremaps = f"{self.PATH}featuremaps/"
        # Object segmentation masks (polished)
        if "objectmasks" in datadict:
            self.objectmasks = datadict["objectmasks"]
        else:
            self.objectmasks = f"{self.PATH}polished_segmentation/"
        # Optical flow
        if "flowmaps" in datadict:
            self.flowmaps = datadict["flowmaps"]
        else:
            self.flowmaps = f"{self.PATH}optical_flow/"
        # Ground truth fixation maps, normalized for calculating NSS
        if "nssmaps" in datadict:
            self.nssmaps = datadict["nssmaps"]
        else:
            self.nssmaps = f"{self.PATH}gt_fov_maps_333/"
        # Original videos in RGB, just used for visualization
        if "videoframes" in datadict:
            self.videoframes = datadict["videoframes"]
        else:
            self.videoframes = f"{self.PATH}videos/"
        # list of videos used for modeling, usually limited by nice segmentations
        if "used_videos" in datadict:
            self.used_videos = datadict["used_videos"]
        else:
            self.used_videos = sorted(
                [name[:-4] for name in os.listdir(self.objectmasks)]
            )
        # Number of frames either given for all or read out based on the segmentation masks
        if "FRAMES_ALL_VIDS" in datadict:
            self.FRAMES_ALL_VIDS = datadict["FRAMES_ALL_VIDS"]
            self.video_frames = {
                video: self.FRAMES_ALL_VIDS for video in self.used_videos
            }
        else:
            self.video_frames = {
                video: np.load(f"{self.objectmasks}{video}.npy").shape[0]
                for video in self.used_videos
            }
        # get the video dimensions from the segmentation masks, if not provided
        if {"VID_SIZE_X", "VID_SIZE_Y"} <= set(datadict):
            self.VID_SIZE_X = datadict["self.VID_SIZE_X"]
            self.VID_SIZE_Y = datadict["self.VID_SIZE_Y"]
        else:
            self.VID_SIZE_Y, self.VID_SIZE_X = np.load(
                f"{self.objectmasks}{self.used_videos[0]}.npy"
            ).shape[1:]
        # For evaluation if gaze point is attributed to an object, defaults to 1 DVA
        if "RADIUS_OBJ_GAZE" in datadict:
            self.RADIUS_OBJ_GAZE = datadict["RADIUS_OBJ_GAZE"]
        else:
            self.RADIUS_OBJ_GAZE = 1.0 * self.DVA_TO_PX
        # check if predefined train or testset is specified
        if "trainset" in datadict:
            assert set(datadict["trainset"]) <= set(self.used_videos)
            self.trainset = datadict["trainset"]
        else:
            self.trainset = self.used_videos
        if "testset" in datadict:
            assert set(datadict["testset"]) <= set(self.used_videos)
            self.testset = datadict["testset"]
        else:
            self.testset = None

        # Check if a path to the ground truth foveation evaluation dataframe is provided.
        # If not, there should be the path to the files and a function to do this evaluation in a class method.
        if "gt_foveation_df" in datadict:
            self.gt_foveation_df = pd.read_pickle(
                self.PATH + datadict["gt_foveation_df"]
            )
            if self.trainset:
                self.train_foveation_df = self.gt_foveation_df[
                    self.gt_foveation_df["video"].isin(self.trainset)
                ]
            else:
                self.train_foveation_df = pd.DataFrame()
            if self.testset:
                self.test_foveation_df = self.gt_foveation_df[
                    self.gt_foveation_df["video"].isin(self.testset)
                ]
            else:
                self.test_foveation_df = pd.DataFrame()

        else:
            assert (
                "eye_tracking_data" in datadict
            ), f"gt_foveation_df or eye_tracking_data needed in dataconfig!"
            self.gt_foveation_df = self.create_foveation_df(
                datadict["eye_tracking_data"]
            )

        # for NSS evaluation, another df has to be provided...
        if "gt_fovframes_nss_df" in datadict:
            self.gt_fovframes_nss_df = pd.read_csv(
                self.PATH + datadict["gt_fovframes_nss_df"],
                usecols=["frame", "x", "y", "subject", "video", "nss"],
            )
        else:
            assert (
                "eye_tracking_data" in datadict
            ), f"GT eye tracking data needed in dataconfig to calculate NSS scores!"
            self.gt_fovframes_nss_df = self.create_nss_df(datadict["eye_tracking_data"])

    def load_yaml(self, path):
        """
        Load a yaml file into a dict.

        :param path: Path to the yaml file.
        :type path: str
        :return: Dictionary with the yaml file content.
        :rtype: dict
        """
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def load_objectmasks(self, videoname):
        """
        Loads the object masks for a given video. It assumes that these masks are stored in objectmasks/videoname as .npy file.

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :return: Object segmentation masks of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        masks = np.load(f"{self.objectmasks}{videoname}.npy")
        assert masks.shape == (
            self.video_frames[videoname],
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
        ), "Segmentation masks are not in shape (f,y,x)!"
        return masks

    def load_featuremaps(self, videoname, featuretype=None, centerbias=None):
        """
        Loads the feature maps for a given video. It assumes that these masks are stored in featuremaps/featuretype/videoname as .npy file.

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :param featuretype: Type of feature maps to load, has to be provided in a subdir, defaults to None
        :type featuretype: str, optional
        :param centerbias: Multiplicative center bias,if `anisotropic_default` it uses anisotropic_centerbias with default params.
                           Otherwise, it must be of shape (VID_SIZE_Y, VID_SIZE_X) and will be used directly, defaults to None
        :type centerbias: np.ndarray or str, optional
        :return: Feature maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        if featuretype in set((None, "None")):
            featuremaps = 0.5 * np.ones(
                (self.video_frames[videoname], self.VID_SIZE_Y, self.VID_SIZE_X)
            )
        else:
            featuremaps = np.load(f"{self.featuremaps}{featuretype}/{videoname}.npy")
            assert featuremaps.shape == (
                self.video_frames[videoname],
                self.VID_SIZE_Y,
                self.VID_SIZE_X,
            ), "Feature maps are not in shape (f,y,x)!"
        if centerbias is None:
            return featuremaps
        elif centerbias == "anisotropic_default":
            return featuremaps * anisotropic_centerbias(
                self.VID_SIZE_X, self.VID_SIZE_Y
            )
        # elif: Implement other keywords?!
        else:
            assert centerbias.shape == (
                self.VID_SIZE_Y,
                self.VID_SIZE_X,
            ), "Provided center bias needs to match the size of the feature map!"
            return featuremaps * centerbias

    def load_flowmaps(self, videoname):
        """
        Loads the optical flow maps for a given video, which have the shape (f-1,y,x,2).
        It assumes that these maps are stored in flowmaps/videoname as .npy file.

        :param videoname: Video for which the OF maps are loaded
        :type videoname: str
        :return: Optical flow maps of shape (frames-1, VID_SIZE_Y, VID_SIZE_X, 2)
        :rtype: np.ndarray
        """
        flowmaps = np.load(f"{self.flowmaps}{videoname}.npy")
        assert flowmaps.shape == (
            self.video_frames[videoname] - 1,
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
            2,
        ), "OF maps are not in shape (f-1,y,x,2)!"
        return flowmaps

    def load_nssmaps(self, videoname):
        """
        Loads the ground truth foveation map for a given video, normalized for calculating the NSS score

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :return: NSS maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        nssmaps = np.load(f"{self.nssmaps}{videoname}.npy")
        assert nssmaps.shape == (
            self.video_frames[videoname],
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
        ), "NSS maps are not in shape (f,y,x)!"
        return nssmaps

    def load_videoframes(self, videoname, ext="mpg"):
        """
        Loads the frames of a given video, only used for visualization.
        Assumes that imageio with ffmpeg is installed.

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :param ext: File extension of the video, defaults to "mpg"
        :type videoname: str
        :return: NSS maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        vid = imageio.get_reader(f"{self.videoframes}{videoname}.{ext}", "ffmpeg")  # type: ignore # format=Format('ffmpeg'))

        vidlist = []
        for image in vid.iter_data():
            vidlist.append(np.array(image))
            # vidlist.append(np.array(resize(image, (540, 960)))) # only needed when using dance01-VIZT.mpg
        del vid
        nframes = self.video_frames[videoname]

        assert len(vidlist) >= nframes, "Number of frames is too small!"
        assert vidlist[0].shape == (
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
            3,
        ), "Frames are not in shape (y,x,3)"

        return vidlist[:nframes]

    def get_fovcat_ratio(self, videos="all"):
        """
        Calculates the ratio of time that the human scanpaths on the given videos spent in each foveation category.

        :param videos: Videos that should be considered (`all`, `train`, `test` or `sgl_vid`), defaults to "all"
        :type videos: str, optional
        :raises Exception: Invalid `videos`
        :return: Dictionary with keys ["B", "D", "I", "R"] and how much time is spent in each
        :rtype: dict
        """
        if videos == "all":
            df_gtfov = self.gt_foveation_df
        elif videos == "train":
            df_gtfov = self.train_foveation_df
        elif videos == "test":
            df_gtfov = self.test_foveation_df
        elif videos in self.used_videos:
            df_gtfov = self.gt_foveation_df[self.gt_foveation_df["video"] == videos]
        else:
            df_gtfov = pd.DataFrame()
            raise Exception(
                f"fovcat can be calcd for `all`, `train`, `test` or `sgl_vid`, you ask for {videos}"
            )

        assert (
            len(df_gtfov) > 0
        ), "`df_gtfov` is empty, make sure that `videos` is a valid input and that its not empty (if test/train)"
        categories = ["B", "D", "I", "R"]
        ratios = {}
        full_dur = np.nansum(df_gtfov.duration_ms)
        for cat in categories:
            ratio = (
                np.nansum(df_gtfov[df_gtfov["fov_category"] == cat].duration_ms)
                / full_dur
            )
            ratios[cat] = ratio
        return ratios

    def get_foveation_ratio(self):
        """
        Returns the ratio of time spent during foveation across the dataset.
        In simulation, this is 1, since saccades are instantaneous and there is no tracker-noise or blinks.
        This affects the detection ratio, since all objects can only be detected once.
        """
        assert (
            len(self.gt_foveation_df) > 0
        ), "`result_df` is empty, make sure to run `evaluate_all_to_df` first!"
        fov_dur = np.sum(self.gt_foveation_df.duration_ms)
        full_dur = 0
        for vid in set(self.gt_foveation_df["video"]):
            nframes = self.video_frames[vid]  # in case some videos are longer
            df_temp = self.gt_foveation_df[self.gt_foveation_df["video"] == vid]
            for s in set(df_temp["subject"]):
                full_dur += nframes / self.FPS * 1000  # unit is ms!
        return fov_dur / full_dur

    #######################################################################
    ##########                NOT YET IMPLEMENTED                ##########
    #######################################################################

    def create_foveation_df(self, eye_tracking_data):
        """
        Will create a dataframe with all ground truth foveations and their statistics
        from the ground truth eye tracking data (what format though?).
        """
        raise NotImplementedError

    def create_nss_df(self, eye_tracking_data):
        """
        Will create a dataframe with all nss scores from the ground truth eye tracking data
        (what format though?).
        """
        raise NotImplementedError
