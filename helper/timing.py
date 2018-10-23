from time import perf_counter
import numpy as np

def nice_s(*seconds):
    """Nice format for seconds with hours and minutes values if needed.

    Example:
        12.2 -> "12.2s"
        78.4 -> "1m 18.400s"
        4000 -> "1h  6m 40.0s"

    Parameters
    ----------
    *seconds: float
        value to convert

    Return
    ------
    single string or list of strings
    """
    if len(seconds) > 1:
        return [nice_s(s) for s in seconds]
    else:
        s = seconds[0]
    if s < 60:
        return "%.3gs" % s
    m, s = divmod(s, 60)
    if m < 60:
        return "%2dm %.3fs" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %2dm %.1fs" % (h, m, s)


class Timing:
    def __init__(self, n):
        self._ticks = [perf_counter(), ]
        self._ticks_diff = []
        self.n = n

    def tic(self):
        self._ticks.append(perf_counter())
        self._ticks_diff.append(self._ticks[-1] - self._ticks[-2])

    @property
    def step_done(self):
        return len(self._ticks_diff)

    @property
    def step_left(self):
        return self.n - self.step_done

    @property
    def time_left(self):
        if self.step_done > 0:
            return np.mean(self._ticks_diff) * self.step_left
        else:
            return 0

    @property
    def time_left_str(self):
        if self.step_done > 0:
            return nice_s(np.mean(self._ticks_diff) * self.step_left)
        else:
            return 'Unknown'

