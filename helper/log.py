# TODO: Use real logger
# for now  use this

DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3


def get_logger(name=None, level=INFO):
    return Log(name, level)


class Log:
    def __init__(self, name, level):
        self._name = name
        self.level = level

        if name is None:
            fstr = ''
        else:
            fstr = name + ': '

        self.msg_fmt = {
            DEBUG: '{}%s'.format(fstr),
            INFO: '{}%s'.format(fstr),
            WARNING: 'WARNING: {}%s'.format(fstr),
            ERROR: '**ERROR**: {}%s'.format(fstr),
        }

    def _log(self, level_msg, mess, *args):
        if self.level > level_msg:
            return
        if len(args) > 0:
            print((self.msg_fmt[level_msg] % mess) % args)
        else:
            print(self.msg_fmt[level_msg] % mess)

    def dbg(self, mess, *args):
        self._log(DEBUG, mess, *args)

    def info(self, mess, *args):
        self._log(INFO, mess, *args)

    def __call__(self, mess, *args):
        self._log(INFO, mess, *args)

    def wrn(self, mess, *args):
        self._log(WARNING, mess, *args)

    def err(self, mess, *args):
        self._log(ERROR, mess, *args)

    def set_to_info(self):
        self.level = INFO

    def set_to_debug(self):
        self.level = DEBUG

    def set_to_warning(self):
        self.level = WARNING

    def set_to_error(self):
        self.level = ERROR

    def set_level(self, level):
        current_level = self.level
        if isinstance(level, str):
            l = level.lower()
        else:
            l = level
        if l in ['e', 'err', 'error', ERROR]:
            self.set_to_error()
        elif l in ['w', 'wrn', 'warning', WARNING]:
            self.set_to_warning()
        elif l in ['i', 'inf', 'info', INFO]:
            self.set_to_info()
        elif l in ['d', 'dbg', 'debug', DEBUG]:
            self.set_to_debug()
        else:
            self.err('Cannot set logging level to %s', level)
        return self.level
