""" Helper functions and class to open and parse files. """

import os.path as path
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ..helper import get_logger
from ..helper import lazy_property

log = get_logger(name=__name__)


class FileParserBase:
    """ Base for file parser.

    Properties
    ----------
    filename: str
        The filename
    is_ok: bool
        If the file was loading correctly.
    messages: lst
        A list of string describing errors
    is_read: bool
        The file had been read.
    """

    def __init__(self, filename: str, common_path=None):
        """ Initialize basic fileparser parameters

        Parameters
        ----------
        filename: str
            The filename
        """
        if common_path is None:
            self.path = filename
            self.filename = filename
        else:
            self.path = path.join(common_path, filename)
            self.filename = filename
        self.is_ok = False

        self.header = self.get_header()

    def get_header(self):
        self.is_ok = True
        return dict()

    @property
    def file(self):
        return open(self.path, 'rb')


class RegexHeaderParser(FileParserBase):
    """ Parse based on some regex expression."""
    header_regex = None
    header_end = None
    dataoffset = None

    def get_header(self):
        """ Read the header from file and parse it. The subclass need to
        define header_regex (re.compile), header_end (string) and
        dataoffet(int). See ColonHeaderFile for example.
        """
        header_regex = self.__class__.header_regex
        header_end = self.__class__.header_end
        dataoffset = self.__class__.dataoffset

        header = super().get_header()
        self.is_ok = False

        check_def = [header_regex, header_end, dataoffset]
        check_name = ["header_regex", "header_end", "dataoffset"]
        if None in check_def:
            for value, name in zip(check_def, check_name):
                cls = self.__class__
                if value is None:
                    log.err("Missing %s definition for %s.", name, cls)
            return None

        messages = []

        self.is_read = True
        try:
            with self.file as f:
                lines = []
                eof = False
                eoh = -1
                while eoh < 0 and not eof:
                    line = f.readline()
                    if len(line) == 0:
                        eof = True
                    else:
                        line = line.decode().strip()
                        if header_end == line:
                            eoh = f.tell()
                        if len(line) > 0:
                            lines.append(line)
            if eof:
                messages.append("EOF reached before EOH")
            else:
                raw = '\n'.join(lines)
                header.update(**{k: v for k, v in header_regex.findall(raw)})
                self.datastart = eoh + dataoffset
                self.is_ok = True

        except Exception as e:
            log.err(str(e))

        return header


class ColonHeaderFile(RegexHeaderParser):
    """ Parser for colon marked Nanonis file (like .sxm file).
    Subclass need to define header_end (str) for header end matching string
    and dataoffet(int) for the data begin offset after the end of header.
     See SxmFile for example.
    """
    header_regex = re.compile(r':(.*?):\n *(.*?)(?=\n:)',
                              flags=re.DOTALL)

    @lazy_property
    def data(self):
        return self.get_data()

    def get_data(self):
        """ Get the data(numpy.array) from file. If self.shape is define,
        the data will have this shape.

         Returns
         -------
         a: np.array
            The data. Return None is error.
         """

        # try to read the file if is not ok (maybe never read)
        if not self.is_ok:
            return None
        try:
            with self.file as f:
                f.seek(self.datastart)
                raw = f.read()
        except Exception as e:
            self.is_ok = False
            log.err(str(e))
            return None
        try:
            shape = self.shape
        except AttributeError as e:
            return np.frombuffer(raw, dtype='>f4')
        else:
            return np.frombuffer(raw, dtype='>f4').reshape(*shape)


class TabHeaderFile(RegexHeaderParser):
    """ Parser for tabbed header Nanonis file (like .dat file).
    Subclass need to define header_end (str) for header end matching string
    and dataoffet(int) for the data begin offset after the end of header.
     See DatFile for example.
    """
    header_regex = re.compile(r'(.*?)\t(.*?)(?=\n)')

    @lazy_property
    def data(self):
        return self.get_data()

    def get_data(self):
        """ Get the data(pandas.DataFrame) from file

         Returns
         -------
         a: pandas.DataFrame
            The data. Return None is error.
         """
        # try to read the file if is not ok (maybe never read)
        if not self.is_ok:
            return None
        try:
            with self.file as f:
                f.seek(self.datastart)
                return pd.read_table(f, sep='\t', )
        except Exception as e:
            self.is_ok = False
            lof.err(str(e))
            return None


class Parse:
    """ Helper function for parsing header."""

    @staticmethod
    def datetime(date, time):
        """ Convert a date and time strings to Python start_time

        Parameters
        ----------
        date: str
            A string in the format day.month.year. Ex.: '23.02.2017'
        time: str
            A string in the format hour:minute:second. Ex.: '23:02:12.32'

        Return
        ------
        The corresponding Datetime
        """
        return datetime.strptime("%s %s" % (date, time), '%d.%m.%Y %H:%M:%S')

    @staticmethod
    def table(s, *types, splitter='\t'):
        """ Read a typed table.
        --> Maybe replace this with call to pandas.read_table

        Parameters
        ----------
        s: str
            A string to parse.
        *types: type
            The type of each columns
        splitter: str
            str to split value

        Return
        ------
        A list a dictionary. Each dictionary is a line with key are the
        name of the columns. Yeah, I know, not the best thing.
        """
        lines = s.strip().split('\n')
        names = [n.strip() for n in lines[0].strip().split('\t')]
        return [dict(zip(names,
                         [t(v) for t, v in
                          zip(types, line.strip().split('\t'))
                          ]
                         )
                     )
                for line in lines[1:]
                ]
