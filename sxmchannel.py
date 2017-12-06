import numpy as np


class SxmChannel:
    def __init__(self, channel, sxm):

        """ Get the raw numpy array of a channel."""
        self.channel_number = sxm.channel_numbers[channel]
        self.channel_name = sxm.channel_names[self.channel_number]
        sxm = self.sxm

        n = self.channel_number

        with sxm.file as sxm:
            sxm.seek(sxm.datastart + self.channel_number * sxm.chunk_size)
            raw = sxm.read(sxm.chunk_size)
            self.data = np.frombuffer(raw, dtype='>f4').reshape(*sxm.shape)

        if sxm.direction == 'up':
            self.data = np.flipud(self.data)
        if n & 0x1:
            self.data = np.fliplr(self.data)
