import sys
import time
import logging
import os
import shutil
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from watchdog.events import FileSystemEventHandler
from watchdog.events import EVENT_TYPE_MODIFIED
from watchdog.events import EVENT_TYPE_MOVED
from watchdog.events import EVENT_TYPE_CREATED
from watchdog.events import EVENT_TYPE_DELETED

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, \
    QFileDialog
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal


class ClonePathEventHandler(LoggingEventHandler):

    """Logs all the events captured."""
    def __init__(self, src, dst, logger ):
        LoggingEventHandler.__init__(self)

        self.src = src
        self.dst = dst

        self.log = logger

        self._method_map = {
            EVENT_TYPE_MODIFIED: self.on_modified,
            EVENT_TYPE_MOVED: self.on_moved,
            EVENT_TYPE_CREATED: self.on_created,
            EVENT_TYPE_DELETED: self.on_deleted,
        }

    def src2dst(self, srcpath):
        return os.path.join(self.dst, srcpath[(len(self.src)+1):])

    def dispatch(self, event):
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
        self._method_map[event.event_type](event)

    def on_moved(self, event):
        super().on_moved(event)

        src = self.src2dst(event.src_path)
        dst = self.src2dst(event.dest_path)

        try:
            self.log("Moving: %s -> %s" % (src, dst))
            shutil.move(src, dst)
        except Exception as e:
            self.log("Exception (it's probably ok) %s" % e)

    def on_created(self, event):
        super().on_created(event)

        what = 'directory' if event.is_directory else 'file'
        src = event.src_path
        dst = self.src2dst(src)
        try:
            if what == 'file':
                self.log("Copying file: %s -> %s" % (src, dst))
                shutil.copy2(src, dst)
            elif what == 'directory':
                self.log("Copying directory: %s -> %s" % (src, dst))
                shutil.copytree(src, dst)
        except Exception as e:
            self.log("Exception (it's probably ok) %s" % e)


    def on_deleted(self, event):
        super().on_deleted(event)

        what = 'directory' if event.is_directory else 'file'
        dst = event.src_path
        self.log("NO deletion of %s" % dst)

    def on_modified(self, event):
        super().on_modified(event)

        what = 'directory' if event.is_directory else 'file'

        src = event.src_path
        dst = self.src2dst(src)

        try:
            if what == 'file':
                self.log("Copying modified file: %s -> %s" % (src, dst))
                shutil.copy2(src, self.src2dst(src))
            elif what =='directory':
                shutil.copytree(src, self.src2dst(src))
                self.log("Copying modified directory: %s -> %s" % (src, dst))
        except Exception as e:
            self.log("Exception (it's probably ok) %s" % e)

class App(QWidget):
    logSignal = pyqtSignal(str, name="Log")

    def __init__(self):
        super().__init__()
        self.title = "Path cloning"
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.src = os.path.abspath(QFileDialog.getExistingDirectory(
            self, "Select source directory"))
        self.dst = os.path.abspath(QFileDialog.getExistingDirectory(
            self, "Select destination directory"))

        self.layout =  QVBoxLayout(self)

        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.move(10, 10)
        self.output.resize(400, 200)
        self.output.appendPlainText("src: %s\ndst: %s" % (self.src, self.dst))
        self.output.setStyleSheet('background-color: rgb(50, 50, 50); color: rgb(200, 200, 200)')

        self.layout.addWidget(self.output, 0)

        self.logSignal.connect(self.log)

        self.show()

    def log(self, event):
        self.output.appendPlainText(str(event))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()

    event_handler = ClonePathEventHandler(ex.src, ex.dst, ex.logSignal.emit)
    #event_handler = LoggingEventHandler()
    observer = Observer()
    observer.schedule(event_handler, ex.src, recursive=True)
    observer.start()

    status = app.exec_()

    observer.stop()
    observer.join()

    sys.exit(status)

