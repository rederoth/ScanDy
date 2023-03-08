import numpy as np
from matplotlib.patches import Polygon


def anisotropic_centerbias(xmax, ymax, sigx2=0.22, v=0.45, mean_to_1=False):
    """
    Function returns a 2D anisotropic Gaussian center bias across the whole frame.
    Inspired by Clark & Tatler 2014 / also used in Nuthmann2017
    Image dimensions are normalized to [-1,1] in x and [-a,a] in y, with aspect ratio a.
    sigx2 is the variance of the Gaussian in x direction, default: 0.22 (Clark&Tatler2014)
    Influence may be smaller (sigx2 bigger) in dynamic scenes (Cristino&Baddeley2009; tHart2009)
    v is a meassure for the anisotropy, default: 0.45 (Clark&Tatler2014 for a=4:3)
    mean_to_1: might make sense to not reduce the overall saliency in each frame.
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
    Function draws a 2D Gaussian on a frame with the dimensions of the video (xmax, ymax).
    x0, y0: center point of the Gaussian
    xsig: standard deviation of the Gaussian, used to be 7.3*DVA_TO_PX (cf. Schwetlick2020)
    ysig: default assumes to be the same as xsig
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
    TODO: This makes the evaluation function much slower! --> find better way!
    """
    if -180 < angle <= 180:
        return angle
    elif angle > 180:
        return angle - 360
    else:
        return angle + 360


def object_at_position(segmentationmap, xpos, ypos, radius=None):
    """
    Function that returns the currently gazed object with a tolerance (radius) around the gaze point.
    If the gaze point is on the background but there are objects within the radius, it is not considered to be background!

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
    # https://stackoverflow.com/questions/39728723/vertical-line-at-the-end-of-a-cdf-histogram-using-matplotlib
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


### Snippet that maps the OF to a nice colormap!
# # Encoding: convert the algorithm's output into Polar coordinates
# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# # Use Hue and Value to encode the Optical Flow
# hsv[..., 0] = ang * 180 / np.pi / 2
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

# # Convert HSV image into BGR for demo
# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow("frame", frame_copy)
# cv2.imshow("optical flow", bgr)
