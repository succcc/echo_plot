import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import scipy.special as sp
import scipy.signal as sg
from scipy import fftpack
from PyQt5 import QtCore, QtGui, QtWidgets
import window
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog, QAbstractItemView
from lmfit import Parameters
from lmfit import Model
import ntpath
import glob
import os.path
import re
import toolFunctions


import ctypes
myappid = u'qdl.python.echoPlot.002' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Main(QMainWindow,window.Ui_MainWindow, toolFunctions.EchoFunctions, toolFunctions.UtilFunc):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self)
        self.dataArray=[]
        self.dataSubArray=[]
        #connect action for button click, checkbox change, doubleclick etc...
        self.openFileBtn.clicked.connect(self.getfile)
        self.clearFileBtn.clicked.connect(self.clearAll)
        self.dataList.itemSelectionChanged.connect(self.selMode)
        self.plotBtn.clicked.connect(self.execPlot)
        self.coarsenlevelSpinBox.valueChanged.connect(self.execPlot)
        self.dataList.itemDoubleClicked.connect(self.execPlot)
        self.smthLengthSpinBox.setSingleStep(2)
        self.smthLengthSpinBox.setMinimum(1)
        self.smthLengthSpinBox.valueChanged.connect(self.execPlot)
        self.smthPolSpinBox.valueChanged.connect(self.execPlot)
        self.smthPolSpinBox.setMinimum(0)
        self.smRadio.setChecked(True)
        self.perpFitRadio.setChecked(True)


        self.delItemBtn.clicked.connect(self.delFromList)

        self.fftBtn.clicked.connect(self.fftPlot)

        self.saveListBtn.clicked.connect(self.saveDataList)
        self.loadListBtn.clicked.connect(self.loadDataList)

        self.dataList.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.echoParamAResultEdit.setReadOnly(True)
        self.echoParamCResultEdit.setReadOnly(True)
        self.echoParamGparaResultEdit.setReadOnly(True)
        self.echoParamNResultEdit.setReadOnly(True)
        self.echoParamFmResultEdit.setReadOnly(True)
        self.echoParamT2ResultEdit.setReadOnly(True)

        # object fig as the plot figure
        self.fig=Figure()
        self.addmpl(self.fig)
        self.directory=None
        self.echoRadio.setChecked(True)
        self.fitCheckBox.clicked.connect(self.enableUpdate)
        if not self.fitCheckBox.isChecked():
            self.besselFitRadio.setEnabled(False)
            self.PLLFitRadio.setEnabled(False)
            self.perpFitRadio.setEnabled(False)
        else:
            self.besselFitRadio.setEnabled(True)
            self.PLLFitRadio.setEnabled(True)
            self.perpFitRadio.setEnabled(True)

        #self.figure = Figure()
        #self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NaviationToolbar(self.canvas, self)
        #self.drawBox = self.mplvl
        #self.drawBox.addWidget(self.canvas)
        #self.canvas.draw()
        #self.addToolBar(self.toolbar)


    def enableUpdate(self):
        if not self.fitCheckBox.isChecked():
            self.besselFitRadio.setEnabled(False)
            self.PLLFitRadio.setEnabled(False)
            self.perpFitRadio.setEnabled(False)
        else:
            self.besselFitRadio.setEnabled(True)
            self.PLLFitRadio.setEnabled(True)
            self.perpFitRadio.setEnabled(True)

    def t2fit(self, x, a, t2, n, d):
        return a * np.exp(-np.power(x / t2, n)) + d

    def resizeEvent(self, event):
        QResizeEvent(event)
        self.tightLayout()

    def tightLayout(self):
        self.fig.tight_layout()

    msize = 5
    lw = 2
    clevel = 2

    #open 'open file dialog' and get the path of data.
    def getfile(self):
        # Case for opening for first time
        if self.directory==None:
            dataPath = QFileDialog.getOpenFileNames(self, "Open file", "D:\Synced_data", "data files (*.*)")
            if not dataPath==([] , ''):
                self.directory=dataPath[0][0]

        # Initial directory as the previous one.
        else:
            dataPath = QFileDialog.getOpenFileNames(self, "Open file", self.directory, "data files (*.*)")
            if not dataPath==([] , ''):
                self.directory=dataPath[0][0]

        #store path and filename
        for dataPath in dataPath[0]:
            filename=os.path.split(dataPath)[1]
            self.addToList(dataPath, filename)

    def saveDataList(self):
        saveFileName = QFileDialog.getSaveFileName(self, "Save list", './saved_list', "list file (*.txt)")[0]
        if not saveFileName == '':
            f = open(saveFileName, 'w')
            for file in self.dataArray:
                f.write(file+'\n')
            f.close()

    def loadDataList(self):
        loadFileName = QFileDialog.getOpenFileName(self, "Load list", './saved_list', "list file (*.txt)")[0]
        if not loadFileName == '':
            with open(loadFileName, 'r') as f:
                direc = f.readlines()
            direc = [x.strip() for x in direc]
            for fnames in direc:
                filename = os.path.split(fnames)[1]
                self.addToList(fnames, filename)

    def delFromList(self):
        idxArray = [item.row() for item in self.dataList.selectionModel().selectedIndexes()]
        idx = self.dataList.currentRow()
        self.dataList.takeItem(idx)
        self.dataArray.pop(idx)

    # automatically select plot mode depending on filename for each case including 'esr', 'rabi', or 'echo'.
    def selMode(self):
        idx = self.dataList.currentRow()
        listSize = self.dataList.count()
        #print("idx: {}, listSize: {}".format(idx, listSize))
        if idx > -1 and listSize > 0:
            if bool(re.search('esr(.+?).txt', self.dataArray[idx])):
                self.esrRadio.setChecked(True)
            if bool(re.search('rabi(.+?).txt', self.dataArray[idx])):
                self.rabiRadio.setChecked(True)
            if bool(re.search('echo(.+?).txt', self.dataArray[idx])):
                self.echoRadio.setChecked(True)

    #--------this is not used---------------
    #split the path into directory and filename
    #head is directory
    #tail is the filename
    def fileDir(self,path):
        head, tail = ntpath.split(path)
        return head

    def fileName(self,path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(path)

    #-----------------------------------------

    #append the data path and filename into Array and dataList.
    #dataArray is just array including path
    #dataList is the Qt object QList which displays the filenames.
    def addToList(self, path, name):
        self.dataArray.append(path)
        self.dataList.addItem(name)

    #clear list and figure
    def clearAll(self):
        self.dataList.clear()
        self.dataArray = []
        self.rmmpl()

    #remove figure
    def rmmpl(self):
        self.mplvlMain.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvlMain.removeWidget(self.toolbar)
        self.removeToolBar(self.toolbar)

    #make figure
    def addmpl(self, fig):
        self.canvas=FigureCanvas(fig)
        self.mplvlMain.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas,
                self, coordinates=True)
        self.addToolBar(self.toolbar)
        #self.cursor = Cursor(self.canvas, useblit=True, color='red', linewidth=2)
        self.canvas.draw()



    def fftPlot(self, fig):
        plt.clf()
        if not self.axe==None:
            print(np.size(self.axe.lines))

            for i in range(np.size(self.axe.lines)):
                plotData=self.axe.lines[i].get_data()
                labelName = self.axe.lines[i].get_label()

                xline = plotData[0]
                N = len(xline)
                #print(xline[1]-xline[0])
                tauInterval = (xline[1]-xline[0])/1e6

                fs = 1/tauInterval
                freqs = fftpack.fftfreq(N) * fs

                yline=plotData[1]
                # print(len(freqs))
                yf = fftpack.fft(yline)
                plt.plot(freqs[0:N//2]/1e6, np.abs(yf[0:N//2]),label=labelName)
            # print(yf)
            plt.xlim(0, fs/1e6/2)
            plt.ylim(-0.1, 1.5)
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Magnitude")
            #plt.legend().draggable()
            plt.show()

    def tauStd(self, datapath, trig='_1_'):
        directory=self.fileDir(datapath)
        normArray = []
        for filename in glob.glob(directory+'/*echo*'+trig+'*.txt'):

            ttau, raws, rawr, tn = np.loadtxt(filename, unpack=True, skiprows=1)
            rawNorm = raws/rawr
            normArray.append(rawNorm)

        if np.ndim(normArray)>1:
            transposed = np.transpose(normArray)
            stdList = []
            for i in range(transposed.shape[0]):
                # print(transposed[i])
                stdE = np.std(transposed[i])
                stdList.append(stdE)
            return stdList
        else:
            stdE = np.std(normArray)
            return stdE

    #actual plotting function
    #all codes are written in this function and do depending on radiobutton selected.
    def execPlot(self):
        #initialize figure
        self.rmmpl()
        self.addmpl(self.fig)

        #index of currently selected item in QList
        idx = self.dataList.currentRow()
        idxArray = [item.row() for item in self.dataList.selectionModel().selectedIndexes()]
        self.coarsenlevel=self.coarsenlevelSpinBox.value()

        if idx > -1 and np.size(self.dataArray) > 0:
            self.axe = self.fig.add_subplot(111)
            self.axe.clear()
            self.axe.autoscale(True)
            #axe.autoscale(tight=True)
            #            fig.canvas.manager.window.raise_()
            #filename = dataPath.split('\\')[1]

            for idx in idxArray:
                dataPath = self.dataArray[idx]
                self.echoPlot(dataPath, yOff=-0.12*idx)


    def echoPlot(self, dataPath, yOff=0):

        filename1_dir, filename1_tail = os.path.split(dataPath)
        if bool(re.search('echo(.+?)_2_(.+?).txt', filename1_tail)):
            filename1_tail = filename1_tail.replace('_2_', '_1_')
        filename1 = filename1_dir + u'/' + filename1_tail
        tau1, rs1, rr1, tn1 = np.loadtxt(filename1, unpack=True, skiprows=1)  # loading first interval

        if bool(re.search('echo(.+?)_1_(.+?).txt', filename1_tail)):
            filename2 = filename1.replace('_1_', '_2_')
            tn1Std = self.tauStd(filename1, trig='_1_')
        else:
            filename2 = 'none_filename2_this_is_not'
            tn1Std = self.tauStd(filename1, trig='')
        tau = tau1
        tn = tn1
        rs = rs1
        rr = rr1
        tnStd = tn1Std

        origTauSize = np.size(tau)
        if os.path.exists(filename2):
            tau2, rs2, rr2, tn2 = np.loadtxt(filename2, unpack=True, skiprows=1)
            tn2Std = self.tauStd(filename1, trig='_2_')
            twoTauStart = tau2[0]
            tau2 = tau2 - twoTauStart
            tau = np.append(tau1, (tau2 + np.max(tau1)))
            tn2_rev = tn2[::-1]
            rs2_rev = rs2[::-1]
            rr2_rev = rr2[::-1]
            tn2Std_rev = tn2Std[::-1]
            tn = np.append(tn1, tn2_rev)
            rs = np.append(rs1, rs2_rev)
            rr = np.append(rr1, rr2_rev)
            tnStd = np.append(tn1Std, tn2Std_rev)
            origTauSize = np.size(tau)

        # We use smoothing 'Savitzky-Golay Filter'
        #int(self.smthLengthSpinBox.text()), int(self.smthPolSpinBox.text())
        tn, rs, rr = self.smooth(tn=tn,
                                 rs=rs,
                                 rr=rr,
                                 smthLength = int(self.smthLengthSpinBox.text()),
                                 smthPol = int(self.smthPolSpinBox.text()),
                                 flag = self.smoothingCheckBox.isChecked())

        # Calling coarsen function
        # coarsen이 나중에 호출된다.
        if self.isCoarsenBox.isChecked():
            if not self.coarsenlevel == 0:
                tau = tau[::self.coarsenlevel]
                tau = tau[0:int(origTauSize / self.coarsenlevel)]
                tnStd = tnStd[0:int(origTauSize / self.coarsenlevel)]
            tn = self.coarsen_array(tn, level=self.coarsenlevel)
            rs = self.coarsen_array(rs, level=self.coarsenlevel)
            rr = self.coarsen_array(rr, level=self.coarsenlevel)
            # print(np.size(tau), np.size(tn))

        # linear space for fitting
        # tau is in [ns] unit. Dividing by 1e9 turns into second unit
        fTx = np.linspace(np.min(tau), np.max(tau)/1e9, 1000)

        # setting fit range, currently the range is sufficiently large to cover original data
        taufIndex = np.where(np.logical_and(tau >= 0, tau <= 10000000))
        taufIndexSize = np.size(taufIndex)
        sizediff = abs(np.size(tn) - taufIndexSize)
        tauf = tau[range(taufIndexSize - sizediff)]
        taufIndexSize = np.size(tn)
        tnf = tn[range(taufIndexSize)]

        self.axe.set_xlabel(r'$2\tau \ (\mu s)$', fontsize=int(self.textSizeEdit.text()))
        self.axe.set_ylabel('PL (A.U.)', fontsize=int(self.textSizeEdit.text()))

        for tick in self.axe.xaxis.get_major_ticks():
            tick.label.set_fontsize(int(self.textSizeEdit.text()))
        for tick in self.axe.yaxis.get_major_ticks():
            tick.label.set_fontsize(int(self.textSizeEdit.text()))

        # fitting and plot implementation
        if self.fitCheckBox.isChecked():
            if self.perpFitRadio.isChecked():
                parrs = Parameters()
                mod = Model(self.prob)
                parrs.add('phi', value=float(self.echoParamPhiStartEdit.text()),
                          min=float(self.echoParamPhiMinEdit.text()),
                          max=float(self.echoParamPhiMaxEdit.text()))
                if self.phiFixCheckBox.isChecked():
                    parrs.add('phi', expr=self.echoParamPhiStartEdit.text())

                parrs.add('n', value=float(self.echoParamNStartEdit.text()),
                          min=float(self.echoParamNMinEdit.text()),
                          max=float(self.echoParamNMaxEdit.text()))
                if self.nFixCheckBox.isChecked():
                    parrs.add('n', expr=self.echoParamNStartEdit.text())

                parrs.add('t2', value=float(self.echoParamT2StartEdit.text()),
                          min=float(self.echoParamT2MinEdit.text()),
                          max=float(self.echoParamT2MaxEdit.text()))
                if self.t2FixCheckBox.isChecked():
                    parrs.add('t2', expr=self.echoParamT2StartEdit.text())

                parrs.add('gPara', value=float(self.echoParamGparaStartEdit.text()),
                          min=float(self.echoParamGparaMinEdit.text()),
                          max=float(self.echoParamGparaMaxEdit.text()))
                if self.gParaFixCheckBox.isChecked():
                    parrs.add('gPara', expr=self.echoParamGparaStartEdit.text())

                parrs.add('gPerp', value=float(self.echoParamGperpStartEdit.text()),
                          min=float(self.echoParamGperpMinEdit.text()),
                          max=float(self.echoParamGperpMaxEdit.text()))
                if self.gPerpFixCheckBox.isChecked():
                    parrs.add('gPerp', expr=self.echoParamGperpStartEdit.text())

                parrs.add('fm', value=float(self.echoParamFmStartEdit.text()),
                          min=float(self.echoParamFmMinEdit.text()),
                          max=float(self.echoParamFmMaxEdit.text()))
                if self.fmFixCheckBox.isChecked():
                    parrs.add('fm', expr=self.echoParamFmStartEdit.text())

                parrs.add('rbz', expr=str(float(self.echoParamGammaBzStartEdit.text())))
                parrs.add('a', value=float(self.echoParamAStartEdit.text()),
                          min=float(self.echoParamAMinEdit.text()),
                          max=float(self.echoParamAMaxEdit.text()))
                if self.aFixCheckBox.isChecked():
                    parrs.add('a', expr=self.echoParamAStartEdit.text())

                parrs.add('c', value=float(self.echoParamCStartEdit.text()),
                          min=float(self.echoParamCMinEdit.text()),
                          max=float(self.echoParamCMaxEdit.text()))
                if self.cFixCheckBox.isChecked():
                    parrs.add('c', expr=self.echoParamCStartEdit.text())

                result = mod.fit(tnf, parrs, t=tauf / 1e9)

                best_a = result.best_values['a']
                best_t2 = result.best_values['t2']
                best_gPara = result.best_values['gPara']
                best_gPerp = result.best_values['gPerp']
                best_fm = result.best_values['fm']
                best_c = result.best_values['c']
                best_phi = result.best_values['phi']
                best_rbz = result.best_values['rbz']
                best_n = result.best_values['n']

                self.textBrowser.setText(result.fit_report())
                self.axe.plot(tau / 1000, tn + yOff, 'o', markersize=int(self.mrkSizeEdit.text()))
                self.axe.plot(fTx * 1e6,
                              self.prob(fTx, t2=best_t2, gPara=best_gPara, gPerp=best_gPerp, fm=best_fm, phi=best_phi,
                                        n=best_n, a=best_a, c=best_c, rbz=best_rbz), 'r-', lw=3)

                self.tightLayout()


                gamma = 2.802  # 2.802 MHz/G
                self.bzEdit.setText("{0:.2f}".format(best_rbz/gamma))
                self.echoParamFmResultEdit.setText("{0:.2f}".format(best_fm))
                self.echoParamGparaResultEdit.setText("{0:.2f}".format(best_gPara))
                self.echoParamGperpResultEdit.setText("{0:.2f}".format(best_gPerp))
                self.echoParamAResultEdit.setText("{0:.2f}".format(best_a))
                self.echoParamCResultEdit.setText("{0:.2f}".format(best_c))
                self.echoParamT2ResultEdit.setText("{0:.2f}".format(best_t2))
                self.echoParamNResultEdit.setText("{0:.2f}".format(best_n))
                self.echoParamPhiResultEdit.setText("{0:.2f}".format(best_phi))

            else:
                # starting Bessel or G_parallel PLL surement
                parrs = Parameters()
                if self.besselFitRadio.isChecked():
                    mod = Model(self.besselFit)
                else:
                    mod = Model(self.pllFit)
                    parrs.add('phi', value=float(self.echoParamPhiStartEdit.text()),
                              min=float(self.echoParamPhiMinEdit.text()),
                              max=float(self.echoParamPhiMaxEdit.text()))
                    if self.phiFixCheckBox.isChecked():
                        parrs.add('phi', expr=self.echoParamPhiStartEdit.text())
                parrs.add('xoff', value=0,
                          min=-1e-4,
                          max=1e-4)
                parrs.add('xoff', expr='0')

                # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                parrs.add('n', value=float(self.echoParamNStartEdit.text()),
                          min=float(self.echoParamNMinEdit.text()),
                          max=float(self.echoParamNMaxEdit.text()))
                parrs.add('t2', value=float(self.echoParamT2StartEdit.text()),
                          min=float(self.echoParamT2MinEdit.text()),
                          max=float(self.echoParamT2MaxEdit.text()))

                parrs.add('gPara', value=float(self.echoParamGparaStartEdit.text()),
                          min=float(self.echoParamGparaMinEdit.text()),
                          max=float(self.echoParamGparaMaxEdit.text()))
                parrs.add('fm', value=float(self.echoParamFmStartEdit.text()),
                          min=float(self.echoParamFmMinEdit.text()),
                          max=float(self.echoParamFmMaxEdit.text()))
                #parrs.add('fm', expr=self.echoParamFmStartEdit.text())

                parrs.add('a', value=float(self.echoParamAStartEdit.text()),
                          min=float(self.echoParamAMinEdit.text()),
                          max=float(self.echoParamAMaxEdit.text()))

                parrs.add('c', value=float(self.echoParamCStartEdit.text()),
                          min=float(self.echoParamCMinEdit.text()),
                          max=float(self.echoParamCMaxEdit.text()))

                result = mod.fit(tnf, parrs, x=tauf/1e9)
                best_a = result.best_values['a']
                best_t2 = result.best_values['t2']
                # t2_stderr = result.params['t2'].stderr
                best_gPara = result.best_values['gPara']
                # p_stderr = result.params['p'].stderr
                best_fm = result.best_values['fm']
                best_c = result.best_values['c']
                if self.PLLFitRadio.isChecked():
                    best_phi = result.best_values['phi']
                best_n = result.best_values['n']
                best_xoff = result.best_values['xoff']

                self.textBrowser.setText(result.fit_report())
                #print(result.fit_report())

                dataLabel=''
                if self.PLLFitRadio.isChecked():
                    dataLabel = r"ϕ = {0:.1f} °".format(best_phi)
                if self.besselFitRadio.isChecked():
                    dataLabel = r"Phase-averaged"

                self.axe.plot(tau / 1000, tn + yOff, 'o', markersize=int(self.mrkSizeEdit.text()), label=dataLabel)
                # self.axe.errorbar(tau / 1000, tn, color='k', yerr=tnStd, fmt='o', markersize=8)

                if self.PLLFitRadio.isChecked():
                    self.axe.plot(fTx * 1e6, self.pllFit(fTx, best_xoff, best_t2, best_gPara, best_fm, best_phi, best_n, best_a,
                                               best_c) + yOff
                             , 'r-', lw=3)
                    # label="T2={0:.2f} us, Gpara={1:.2f} MHz".format(best_t2, best_gPara))
                    self.echoParamPhiResultEdit.setText("{0:.2f}".format(best_phi))
                if self.besselFitRadio.isChecked():
                    self.axe.plot(fTx * 1e6, self.besselFit(fTx, best_xoff, best_t2, best_gPara, best_fm, best_n, best_a,
                                                  best_c) + yOff
                             , 'r-', lw=3)
                    # label="T2={0:.2f} us, Gpara={1:.2f} MHz".format(best_t2, best_gPara))
                self.tightLayout()

                self.echoParamFmResultEdit.setText("{0:.2f}".format(best_fm))
                self.echoParamGparaResultEdit.setText("{0:.2f}".format(best_gPara))
                self.echoParamAResultEdit.setText("{0:.2f}".format(best_a))
                self.echoParamT2ResultEdit.setText("{0:.2f}".format(best_t2))
                self.echoParamNResultEdit.setText("{0:.2f}".format(best_n))
                self.axe.legend()

        # plotting echo signal with exponential decay compensated.
        else:
            # exec without fitting
            self.axe.plot(tau / 1000, tn + yOff, 'o', markersize=int(self.mrkSizeEdit.text()), label=filename1_tail)
        self.axe.legend(fontsize = 10, bbox_to_anchor=(0.98, 1), borderaxespad=0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Main()
    mainWindow.setWindowTitle('Strain-echo Plotter')
    mainWindow.setWindowIcon(QIcon('strainEcho.png'))
    mainWindow.show()

    sys.exit(app.exec_())
