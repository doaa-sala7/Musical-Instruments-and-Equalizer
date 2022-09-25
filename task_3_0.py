import sys
import sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QMessageBox
from scipy.io.wavfile import read
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid import make_axes_locatable
from threading import Thread
from task3_1edited import Ui_MainWindow
from pyqtgraph.examples.optics import ParamObj
import playsound
# import phonon
import winsound
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl


class ApplicationWindow(QtWidgets.QMainWindow):
    dataSet = []
    volume = 1
    isPaused = False
    step = 0
    stepRight = 0
    gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    MIN_FREQ = [0.1, 2000, 5500]
    MAX_FREQ = [2000, 5500, 10500]
    path="E:\SBME\SBME 3rd\1st term\DSP\Tasks\task3\يارب\يارب"


    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.arr = [self.ui.drums, self.ui.piano, self.ui.saxophone, self.ui.violin, self.ui.Xylophone]
        for n in range(len(self.arr)):
            self.connect(n)
        self.ui.load.clicked.connect(self.loadFile)
        self.ui.play.clicked.connect(self.playPause)
        self.player = QMediaPlayer()
        url = QUrl.fromLocalFile(self.path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()
        # self.ui.horizontalSlider_volume.valueChanged.connect(self.player.setVolume)
        self.ui.horizontalSlider_volume.valueChanged.connect(lambda: self.Set_Volume())

        # self.ui.volum.sliderReleased.connect(self.Volume)
        window = self.ui.tabWidget

        # white buttons
        self.ui.PW1.clicked.connect(lambda: self.PW1())
        self.ui.PW2.clicked.connect(lambda: self.PW2())
        self.ui.PW3.clicked.connect(lambda: self.PW3())
        self.ui.PW4.clicked.connect(lambda: self.PW4())
        self.ui.PW5.clicked.connect(lambda: self.PW5())
        self.ui.PW6.clicked.connect(lambda: self.PW6())
        self.ui.PW7.clicked.connect(lambda: self.PW7())
        # black buttons
        self.ui.PB1.clicked.connect(lambda: self.PB1())
        self.ui.PB2.clicked.connect(lambda: self.PB2())
        self.ui.PB3.clicked.connect(lambda: self.PB3())
        self.ui.PB4.clicked.connect(lambda: self.PB4())
        self.ui.PB5.clicked.connect(lambda: self.PB5())
        # Drums
        self.ui.D1.clicked.connect(lambda: self.D1())
        self.ui.D2.clicked.connect(lambda: self.D2())
        # Guitar
        self.ui.G1.clicked.connect(lambda: self.G1())
        self.ui.G2.clicked.connect(lambda: self.G2())
        self.ui.G3.clicked.connect(lambda: self.G3())
        self.ui.G4.clicked.connect(lambda: self.G4())

    # SimplePianoMusic
    def PW1(self):
        winsound.PlaySound('D', winsound.SND_FILENAME)

    def PW2(self):
        winsound.PlaySound('G', winsound.SND_FILENAME)

    def PW3(self):
        winsound.PlaySound('B', winsound.SND_FILENAME)

    def PW4(self):
        winsound.PlaySound('E', winsound.SND_FILENAME)

    def PW5(self):
        winsound.PlaySound('C', winsound.SND_FILENAME)

    def PW6(self):
        winsound.PlaySound('F', winsound.SND_FILENAME)

    def PW7(self):
        winsound.PlaySound('D1', winsound.SND_FILENAME)

    # SharpPianoMusic
    def PB1(self):
        winsound.PlaySound('G_s', winsound.SND_FILENAME)

    def PB2(self):
        winsound.PlaySound('D_s', winsound.SND_FILENAME)

    def PB3(self):
        winsound.PlaySound('F_s', winsound.SND_FILENAME)

    def PB4(self):
        winsound.PlaySound('C_s', winsound.SND_FILENAME)

    def PB5(self):
        winsound.PlaySound('D_s1', winsound.SND_FILENAME)

    # DrumsMusic
    def D1(self):
        winsound.PlaySound('DOM', winsound.SND_FILENAME)

    def D2(self):
        winsound.PlaySound('DOMM', winsound.SND_FILENAME)

    # GuitarMusic
    def G1(self):
        winsound.PlaySound('GUITAR1', winsound.SND_FILENAME)

    def G2(self):
        winsound.PlaySound('GUITAR2', winsound.SND_FILENAME)

    def G3(self):
        winsound.PlaySound('GUITAR3', winsound.SND_FILENAME)

    def G4(self):
        winsound.PlaySound('GUITAR4', winsound.SND_FILENAME)

    def loadFile(self):
        self.ui.mainGraph.clear()
        self.firstPoint = 0
        self.fname, self.format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Signal File", "")
        self.fname = self.fname.split('/')[-1]
        self.format = self.fname.split('.')[-1]
        if self.fname == "":
            pass
        else:
            if self.format != "wav":
                pass
            else:
                self.rate, self.dataSet = read(self.fname)
                self.duration = len(self.dataSet) / self.rate
                self.reset()
                self.time = np.arange(0, self.duration, 1 / self.rate)
                self.zeros = np.zeros(self.dataSet.size)
                self.original = np.fft.fft(self.dataSet)
                self.newsignal = np.copy(self.original)
                self.fft_fre = np.fft.fftfreq(n=len(self.dataSet), d=1 / self.rate)
                self.freq_bins = int(len(self.dataSet) * 0.5)
                self.spectrogram(self.original)
                self.ui.mainGraph.plot(self.time, self.dataSet, pen=[216, 191, 216])
                self.step = 0




    def start(self):
        if self.stepRight == 0:
            self.ui.timer.timeout.connect(self.updateData)
        self.isPaused = False
        self.ui.timer.start(100)

    def spectrogram(self, frequency):
        self.ui.axes.clear()
        self.ui.axes.specgram(frequency, Fs=self.rate, cmap='magma')
        self.ui.figure.colorbar(mpl.cm.ScalarMappable(norm=self.ui.norm, cmap='magma'), cax=self.ui.c_bar_ax,
                                orientation='horizontal')
        self.ui.canvas.draw()

    def playPause(self):
        self.update(False)
        if self.ui.play.text() == "play":
            self.start()
            self.norm_value()
            sd.play(self.dataSet[int(self.firstPoint):], self.rate)
            self.ui.play.setText("pause")
        else:
            self.isPaused = True
            sd.stop()
            self.ui.play.setText("play")

    def reset(self):
        self.step = 0
        sd.stop()
        self.ui.mainGraph.setXRange(0, self.duration)
        self.ui.timer.stop()
        if self.ui.play.text() == "pause":
            self.ui.play.setText("play")

    def update(self, bol_play=True):
        self.x_mid = (self.step + self.stepRight) / 2
        self.starter_point = (self.x_mid / self.duration) * len(self.dataSet)
        if (bol_play):
            sd.play(self.dataSet[int(self.firstPoint):], self.rate)

    def updateData(self):
        if not self.isPaused:
            self.stepRight = self.step + 2
            self.step += 0.1
            self.ui.mainGraph.plotItem.setXRange(self.step, self.stepRight)
            if int(self.step) == int(self.time[-1]):
                self.ui.timer.stop()

    def equalizer(self, indx, gain):
        gain = gain + 0.001
        min_freq = self.MIN_FREQ[indx]
        max_freq = self.MAX_FREQ[indx]
        level = indx
        first = np.where(self.fft_fre == min_freq)[0][0]
        last = np.where(self.fft_fre == max_freq)[0][0]
        self.newsignal[first:last] = self.newsignal[first:last] / self.gains[level]
        self.gains[level] = gain
        self.newsignal[first:last] = self.newsignal[first:last] * self.gains[level]
        self.spectrogram(self.newsignal)
        bol_play = True
        self.dataSet = np.fft.ifft(self.newsignal)
        self.dataSet = self.dataSet.real
        self.ui.mainGraph.clear()
        self.dataSet = np.ascontiguousarray(self.dataSet, dtype=np.int32)
        self.norm_value()
        if self.ui.play.text() == "play":
            bol_play = False
        self.update(bol_play)
        self.ui.mainGraph.plot(self.time, self.dataSet, pen=[216, 191, 216])

    def volumeControl(self):
       self.player.setVolume(self.ui.horizontalSlider_volume.value())


    def connect(self, index):
        self.arr[index].sliderReleased.connect(lambda: self.sliders(index))

    def sliders(self, index):
        self.value = self.arr[index].value()
        self.equalizer(index, self.value)

    def norm_value(self):
        self.dataNorm = self.dataSet / self.dataSet.max()
        self.dataSet = self.volume * self.dataNorm

    def Set_Volume(self):
        bol_play = True
        self.volume = self.ui.horizontalSlider_volume.value()/100
        if self.volume == 0:
            sd.play(self.zeros, self.rate)
        else:

            self.norm_value()
            if self.ui.play.text() == "PLAY":
                bol_play = False
            self.update(bol_play)

    def show_popup(self, message, information):
        msg = QMessageBox()
        msg.setWindowTitle("Message")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.setInformativeText(information)
        msg.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
