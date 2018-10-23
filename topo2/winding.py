



from time import perf_counter

def get_value(self, xy, position='relative'):
    """ The real position (nm) of all pixels as a lazy property. """


    # position of (x,y) in pixel space
    pospx = self.realpos_to_pxpos(xy, position=position)

    # corresponding pixel
    px = np.floor(pospx).astype(int)

    # unit square for the position in pixel space
    sq = np.array([pospx + (i, j) for i in (0, 1) for j in (0, 1)])

    # corresponding pixels
    sq_px = np.floor(sq).astype(int)

    # fraction of each pixels in the square
    sq_frac = np.abs(np.prod(np.floor(sq[3]) - sq, axis=1))

    # corner case
    k = sq_frac > 0
    if np.any(~k):
        return np.sum(self.data[sq_px[k, 0], sq_px[k, 1]] * sq_frac[k])
    else:
        return np.sum(self.data[sq_px[:, 0], sq_px[:, 1]] * sq_frac)


def is_px_in_circle(self, c_xy, c_r, c_theta=None, c_arc=np.pi):
    c_xy = np.array(c_xy)
    realpos = self.realpos_of_all_pixels
    are_in_circle = np.zeros(self.size_px, dtype=bool)

    limit = np.array([
        np.floor((c_xy - c_r) * self.conv_real_to_px),
        np.ceil((c_xy + c_r) * self.conv_real_to_px),
    ]).astype(int)

    are_in_circle[limit[0, 0]:limit[1, 0], limit[0, 1]:limit[1, 1]] = True
    square = realpos[are_in_circle]
    square = np.sum(np.square(square - c_xy), axis=1) < c_r * c_r
    are_in_circle[are_in_circle] = square

    # are_in_circle = np.sum(np.square(realpos - c_xy), axis=2) < c_r * c_r

    # make sure that c_theta is between -pi and pi
    if c_theta is not None:
        # relative position of the points
        are_in = realpos[are_in_circle] - c_xy

        # angles of the points
        angles = np.arctan2(are_in[:, 1], are_in[:, 0]) - c_theta
        # replace the points in -pi..pi
        angles = angles - 2 * np.pi * np.ceil((angles // np.pi) / 2)

        # angles = angles - theta

        if c_arc > np.pi:
            are_in = (angles > 0) | (angles < c_arc - 2 * np.pi)
        else:
            are_in = (angles > 0) & (angles < c_arc)

        are_in_circle[are_in_circle] = are_in

    return are_in_circle

def px_in_circle(self, c_xy, c_r, c_theta=None, c_arc=None):
    return self.data[self.is_px_in_circle(c_xy, c_r, c_theta, c_arc)]

def get_winding_circle(self, points=None, c_radius=0.44, c_arc=np.pi,
                       search_steps=(30, 30), step_zoom=0.1,
                       search_start=0.0,
                       show_estimate=True
                       ):
    if points is None:
        points = self.realpos_of_all_pixels

    pdim = len(points.shape)

    def f_angle(p, angle):
        a = self.px_in_circle(p, c_radius, angle, c_arc)
        b = self.px_in_circle(p, c_radius, angle, c_arc + np.pi)
        if len(a) == 0 or len(b) == 0:
            if len(a) == 0:
                print("Empty circle (a) for point %s at angle %s" %
                      (p, angle))
            if len(b) == 0:
                print("Empty circle (b) for point %s at angle %s" %
                      (p, angle))
            return 0.0
        return a.mean() - b.mean()

    def f_best(p):
        vmin = search_start
        vmax = search_start + 2 * np.pi

        for steps in search_steps:
            xv = np.linspace(vmin, vmax, steps, endpoint=False)
            yv = np.array([f_angle(p, x) for x in xv])
            angle = xv[yv.argmax()]
            magnitude = yv.max()
            r = (vmax - vmin) * step_zoom
            vmin = angle - r
            vmax = angle + r

        return angle, magnitude

    s = points.shape
    tot = np.prod(s[:-1])
    rs = (tot, s[-1])
    est = 0.1 * rs[0]
    pts = points.reshape(rs)
    out = np.empty(rs)
    t0 = perf_counter()
    for i in range(tot):
        out[i] = f_best(pts[i])

        if show_estimate and i > est:
            t1 = perf_counter()
            elapsed = t1 - t0
            per_point = elapsed / (i + 1)
            estimated_total = per_point * tot

            print('Time per point: %.5f, Estimated total: %.3fs' % (
                per_point, estimated_total))
            show_estimate = False

    return out.reshape(s)
