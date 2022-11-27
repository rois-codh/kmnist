import sys
import logging
import os

def get_logger(exp_dir):
    """
    creates logger instance. writing out info to file and to terminal.
    :param exp_dir: experiment directory, where exec.log file is stored.
    :return: logger instance.
    """
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    logger = logging.getLogger('ChestXray_detection')
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(exp_dir, 'log.txt')
    hdlr = logging.FileHandler(log_file)
    print('Logging to {}'.format(log_file))
    logger.addHandler(hdlr)
    logger.addHandler(ColorHandler())
    logger.propagate = False
    return logger

class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37, default=39)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))


class ColorHandler(logging.StreamHandler):

    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "default",
            logging.WARNING: "red",
            logging.ERROR: "red"
        }
        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(record.msg + "\n", color)
