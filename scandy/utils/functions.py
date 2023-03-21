import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

from matplotlib.patches import Polygon
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution


def anisotropic_centerbias(xmax, ymax, sigx2=0.22, v=0.45, mean_to_1=False):
    """
    Function returns a 2D anisotropic Gaussian center bias across the whole frame.
    Inspired by Clark & Tatler 2014 / also used in Nuthmann2017
    Image dimensions are normalized to [-1,1] in x and [-a,a] in y, with aspect ratio a.
    The default values have been taken from Clark & Tatler (2014).
    Influence may be smaller (sigx2 bigger) in dynamic scenes (Cristino&Baddeley2009; tHart2009).

    :param xmax: Size of the frame in x-direction
    :type xmax: int
    :param ymax: Size of the frame in y-direction
    :type ymax: int
    :param sigx2: Normalized variance of the Gaussian in x direction, defaults to 0.22
    :type sigx2: float, optional
    :param v: Anisotropy, defaults to 0.45
    :type v: float, optional
    :param mean_to_1: Normalize such that the mean of the Gauss is one (instead of the max), defaults to False
    :type mean_to_1: bool, optional
    :return: Center bias in the form of a 2D Gaussian with dimensions of a frame
    :rtype: np.array
    """
    X, Y = np.meshgrid(
        np.linspace(-1, 1, xmax), np.linspace(-ymax / xmax, ymax / xmax, ymax)
    )
    G = np.exp(-(X**2 / (2 * sigx2) + Y**2 / (2 * sigx2 * v)))
    if mean_to_1:
        G = G / np.mean(G)
    return G


def gaussian_2d(x0, y0, xmax, ymax, xsig, ysig=None, sumnorm=False):
    """
    This function returns a 2D Gaussian around the (current gaze) position (x0,y0)
    on a frame of size (xmax, ymax). If no ysig is provided, the Gaussian is isotropic.

    :param x0: Center point of the Gaussian in x direction
    :type x0: int
    :param y0: Center point of the Gaussian in y direction
    :type y0: int
    :param xmax: Size of the frame in x-direction
    :type xmax: int
    :param ymax: Size of the frame in y-direction
    :type ymax: int
    :param xsig: Standard deviation of the Gaussian in x direction
    :type xsig: float
    :param ysig: If anisotropic, provide std in y direction, defaults to None
    :type ysig: float, optional
    :param sumnorm: Normalize such that mean (instead of max) is one, defaults to False
    :type sumnorm: bool, optional
    :return: 2D Gaussian on a frame
    :rtype: np.array
    """
    if not ysig:
        ysig = xsig
    X, Y = np.meshgrid(np.arange(0, xmax), np.arange(0, ymax))
    G = np.exp(-0.5 * (((X - x0) / xsig) ** 2 + ((Y - y0) / ysig) ** 2))
    if sumnorm:
        return G / np.sum(G)
    else:
        return G


def angle_limits(angle):
    """
    Makes sure that a given angle is within the range of -180<angle<=180

    :param angle: Angle to be tested / converted
    :type angle: float
    :return: angle with -180 < angle <= 180
    :rtype: float
    """
    if -180 < angle <= 180:
        return angle
    elif angle > 180:
        return angle - 360
    else:
        return angle + 360


def object_at_position(segmentationmap, xpos, ypos, radius=None):
    """
    Function that returns the currently gazed object with a tolerance (radius)
    around the gaze point. If the gaze point is on the background but there are
    objects within the radius, it is not considered to be background.

    :param segmentationmap: Object segmentation of the current frame
    :type segmentationmap: np.array
    :param xpos: Gaze position in x direction
    :type xpos: int
    :param ypos: Gaze position in y direction
    :type ypos: int
    :param radius: Tolerance radius, objects within that distance of the gaze point
        are considered to be foveated, defaults to None
    :type radius: float, optional
    :return: Name of the object(s) at the given position / within the radius
    :rtype: str
    """
    (h, w) = segmentationmap.shape
    if radius == None:
        objid = segmentationmap[ypos, xpos]
        if objid == 0:
            objname = "Ground"
        else:
            objname = f"Object {objid}"
        return objname
    # more interesting case: check in radius!
    else:
        center_objid = segmentationmap[ypos, xpos]
        if center_objid > 0:
            return f"Object {center_objid}"
        # check if all in rectangle is ground, then no need to draw a circle
        elif (
            np.sum(
                segmentationmap[
                    max(0, int(ypos - radius)) : min(h - 1, int(ypos + radius)),
                    max(0, int(xpos - radius)) : min(w - 1, int(xpos + radius)),
                ]
            )
            == 0
        ):
            return "Ground"
        # Do computationally more demanding check for a radius
        # store all objects other than `Ground` that lie within the radius
        else:
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - xpos) ** 2 + (Y - ypos) ** 2)
            mask = dist_from_center <= radius
            objects = np.unique(mask * segmentationmap)
            if len(objects) == 1 and 0 in objects:
                return "Ground"
            else:
                return ", ".join([f"Object {obj}" for obj in objects if (obj > 0)])


def fix_hist_step_vertical_line_at_end(ax):
    """
    Get rid of vertical lines on the right of the histograms, as proposed here:
    https://stackoverflow.com/questions/39728723/vertical-line-at-the-end-of-a-cdf-histogram-using-matplotlib

    :param ax: Axis to be fixed
    :type ax: matplotlib.axes._subplots.AxesSubplot
    """
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def plot_var_pars(model, res_path, runid, parameters, par_sym, relative_par_vals):
    """
    Make a summary plot of a parameter exploration.

    This is very specific to the evaluation we do in the supplement of the paper.

    :param model: _description_
    :type model: _type_
    :param res_path: _description_
    :type res_path: _type_
    :param runid: _description_
    :type runid: _type_
    :param parameters: _description_
    :type parameters: _type_
    :param par_sym: _description_
    :type par_sym: _type_
    :param relative_par_vals: _description_
    :type relative_par_vals: _type_
    """

    # load evolution results
    DILLNAME = f"{runid}.dill"
    evol = Evolution(lambda x: x, ParameterSpace(["mock"], [[0, 1]]))
    evol = evol.loadEvolution(res_path + f"{runid}/{DILLNAME}")
    df_evol = evol.dfEvolution(outputs=True).copy()

    # Get mean parameters of last generation
    df_top32 = df_evol.sort_values("score", ascending=False)[:32]
    mean_pars = {}
    for par in parameters:
        mean_pars[par] = np.mean(df_top32[par])
    print(mean_pars)
    gt_amp_dva = model.Dataset.test_foveation_df["sac_amp_dva"].dropna().values
    gt_dur_ms = model.Dataset.test_foveation_df["duration_ms"].dropna().values

    # pre computed values for different factors
    d_res = {}
    for var_par in parameters:
        for factor in relative_par_vals:
            name = f"res_df_mean_{var_par}_{factor}"
            key = f"{var_par}_{factor}"
            d_res[key] = {}
            model.result_df = pd.read_csv(res_path + f"{runid}/{name}.csv")
            sim_dur_ms = model.result_df["duration_ms"].dropna().values
            sim_amp_dva = model.result_df["sac_amp_dva"].dropna().values
            d_res[key]["sim_dur_ms"] = sim_dur_ms
            d_res[key]["sim_amp_dva"] = sim_amp_dva
            if not sim_amp_dva.size:  # if there are no saccades,
                sim_amp_dva = [0]  # this leads to worst fitness...
            ks_amp, p_amp = stats.ks_2samp(gt_amp_dva, sim_amp_dva)
            ks_dur, p_dur = stats.ks_2samp(gt_dur_ms, sim_dur_ms)
            d_res[key]["ks_amp"] = ks_amp
            d_res[key]["ks_dur"] = ks_dur
            d_res[key]["fov_cat"] = list(model.get_fovcat_ratio().values())
    fig, axs = plt.subplots(4, len(parameters), figsize=(11, 11), dpi=200, sharey="row")
    for p, var_par in enumerate(parameters):
        # initialize lists for each parameter, entries are results for different factors
        dur_data = []
        amp_data = []
        f_amps = []
        f_durs = []
        d_fovcat = {}
        for fovcat in ["B", "D", "I", "R"]:
            d_fovcat[fovcat] = []
        # go through factors...
        for f, factor in enumerate(relative_par_vals):
            key = f"{var_par}_{factor}"
            dur_data.append(np.log10(d_res[key]["sim_dur_ms"]))
            amp_data.append(d_res[key]["sim_amp_dva"])
            f_amps.append(d_res[key]["ks_amp"])
            f_durs.append(d_res[key]["ks_dur"])
            for i, fovcat in enumerate(["B", "D", "I", "R"]):
                d_fovcat[fovcat].append(d_res[key]["fov_cat"][i])

        # plotting, parameters are different colums, rows are metrics
        x_par_vals = mean_pars[var_par] * np.array(relative_par_vals)
        if var_par == "ior_inobj":
            x_par_vals = np.clip(x_par_vals, 0, 1)
        axs[0, p].boxplot(dur_data)  # , x='Factor', y='Foveation duration [ms]')
        axs[0, p].set(
            xticklabels=relative_par_vals,
            xlabel=f"Factor for {par_sym[p]}",  # title=f'{var_par}',
            yticks=[1, 2, 3, 4],
            yticklabels=[10, 100, 1000, 10000],
        )
        if p == 0:
            axs[0, p].set_ylabel("Foveation duration [ms]")
        axs[1, p].boxplot(amp_data)  # , x='Factor', y='Saccade amplitude [dva]')
        axs[1, p].set(
            xticklabels=relative_par_vals,
            xlabel=f"Factor for {par_sym[p]}",  # title=f'{var_par}',
            yticks=[0, 10, 20, 30, 40, 50],
            ylim=[0, 50],
        )
        if p == 0:
            axs[1, p].set_ylabel("Saccade amplitude [dva]")
        for fovcat in ["B", "D", "I", "R"]:
            axs[3, p].plot(x_par_vals, d_fovcat[fovcat], "o-", label=fovcat)
        axs[2, p].plot(x_par_vals, f_amps, "o-", label="Sac. amp.")
        axs[2, p].plot(x_par_vals, f_durs, "o-", label="Fov. dur.")
        axs[2, p].axvline(mean_pars[var_par], ls="--", color="k")
        if p == 4:
            axs[2, p].legend()
        axs[2, p].set(xlabel=f"{par_sym[p]}")  # , #title=f'{var_par}',ylim=[0,0.45])
        if p == 0:
            axs[2, p].set_ylabel("Kolmogorov-Smirnov\ndistance")
        if p == 4:
            axs[3, p].legend()
        axs[3, p].set(xlabel=f"{par_sym[p]}")  # , #title=f'{var_par}',ylim=[0,0.5])
        if p == 0:
            axs[3, p].set_ylabel("Fraction of time spent\nper category")
        axs[3, p].axvline(mean_pars[var_par], ls="--", color="k")

    plt.tight_layout()
    sns.despine()
    plt.show()
