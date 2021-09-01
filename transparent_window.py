import sys
import cv2 as cv
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage
from PIL import Image, ImageDraw, ImageQt


def log_uncaught_exceptions(ex_cls, ex, tb):
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    import traceback
    text += ''.join(traceback.format_tb(tb))

    print(text)
    QMessageBox.critical(None, 'Error', text)
    quit()


sys.excepthook = log_uncaught_exceptions


class TransparentWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def show_image(self, image):
        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGRA2RGBA))

        image_qt = ImageQt.toqimage(pil_image)

        pixmap = QPixmap.fromImage(image_qt)

        palette = self.palette()
        palette.setBrush(QPalette.Normal, QPalette.Window, QBrush(pixmap))
        palette.setBrush(QPalette.Inactive, QPalette.Window, QBrush(pixmap))

        self.setPalette(palette)
        self.setMask(pixmap.mask())

    def drag_window(self, event):
        delta = QPoint(event.globalPos() - self.old_position)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.old_position = event.globalPos()

    def mousePressEvent(self, event):
        self.old_position = event.globalPos()
        self.old_position = event.globalPos()

    def mouseMoveEvent(self, event):
        self.drag_window(event)
        x, y = self.get_window_coordinates()
        width, height = self.get_window_size()
        region = x, y, width, height

    def get_window_size(self):
        size = self.frameSize().width(), self.frameSize().height()
        return size

    def get_window_coordinates(self):
        coordinates = self.x(), self.y()
        return coordinates


def main():
    application = QApplication(sys.argv)
    transparent_window = TransparentWindow()
    transparent_window.show()
    sys.exit(application.exec_())


if __name__ == '__main__':
    main()
