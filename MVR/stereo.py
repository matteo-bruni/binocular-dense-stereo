from numpy import zeros, roll, argmax
from numpy.ma import sqrt
from scipy.ndimage import filters


def plane_sweep_ncc(im_l, im_r, start, steps, wid):
    """ Find disparity image using normalized cross-correlation. """
    m,n = im_l.shape
    # arrays to hold the different sums
    mean_l = zeros((m, n))
    mean_r = zeros((m, n))
    s = zeros((m, n))
    s_l = zeros((m, n))
    s_r = zeros((m, n))
    # array to hold depth planes
    dmaps = zeros((m, n, steps))
    # compute mean of patch
    filters.uniform_filter(im_l, wid, mean_l)
    filters.uniform_filter(im_r, wid, mean_r)
    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # try different disparities
    for displ in range(steps):
        # move left image to the right, compute sums
        filters.uniform_filter(roll(norm_l,-displ-start)*norm_r,wid,s) # sum nominator
        filters.uniform_filter(roll(norm_l,-displ-start)*roll(norm_l,-displ-start),wid, s_l)
        filters.uniform_filter(norm_r*norm_r,wid,s_r) # sum denominator

        # store ncc scores
        dmaps[:,:,displ] = s/sqrt(s_l*s_r)

    # pick best depth for each pixel
    return argmax(dmaps, axis=2)


def plane_sweep_gauss(im_l, im_r, start, steps, wid):
    """ Find disparity image using normalized cross-correlation
    with Gaussian weighted neigborhoods. """
    m, n = im_l.shape
    # arrays to hold the different sums
    mean_l = zeros((m, n))
    mean_r = zeros((m, n))
    s = zeros((m, n))
    s_l = zeros((m, n))
    s_r = zeros((m, n))
    # array to hold depth planes
    dmaps = zeros((m, n, steps))
    # compute mean
    filters.gaussian_filter(im_l,wid,0,mean_l)
    filters.gaussian_filter(im_r,wid,0,mean_r)
    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # try different disparities
    for displ in range(steps):
        # move left image to the right, compute sums
        filters.gaussian_filter(roll(norm_l, -displ-start)*norm_r,wid,0,s) # sum nominator
        filters.gaussian_filter(roll(norm_l, -displ-start)*roll(norm_l, -displ-start), wid, 0, s_l)
        filters.gaussian_filter(norm_r*norm_r, wid, 0, s_r) # sum denominator

        # store ncc scores
        dmaps[:, :, displ] = s/sqrt(s_l*s_r)

    # pick best depth for each pixel
    return argmax(dmaps,axis=2)


