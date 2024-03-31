
import tkinter as tk
import ttkbootstrap as tb
from tkinter import *  # filedialog, Text, Label, Entry, Button
from tkinter import messagebox
from tkinter import filedialog
from ttkbootstrap.constants import *

from PIL import ImageTk, Image

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from scipy import signal 
import time as timer
import datetime 
import os
import sys

import threading
from threading import Event
from multiprocessing import Process
from multiprocessing.managers import BaseManager

##################################################
# Global variables
##################################################
# System and Window ontrol
global screenWidth, screenHeight, banner, flags, nrows, ncolumns
global win, thd, disableAnimate
# Widgets

class SignalDection:
    def __init__(self, currentDeviceIndex, currentDeviceName, sampleFrequency, frequency, frequencyDelta, sampleSize, samplesToProcess):
        self.currentDeviceIndex = currentDeviceIndex
        self.currentDeviceName = currentDeviceName
        self.sampleFrequency = sampleFrequency
        self.frequency = frequency
        self.frequencyDelta = frequencyDelta
        self.sampleSize = sampleSize
        self.samplesToProcess = samplesToProcess

        self.timeBuffer = np.zeros((sampleSize,), dtype=float)
        self.amplitudeSine = np.zeros((sampleSize,), dtype=float)
        self.amplitudeCosine = np.zeros((sampleSize,), dtype=float)

        self.amplitudeSN = np.zeros((sampleSize,), dtype=float)
        self.bpFilteredSN = np.zeros((sampleSize,), dtype=float)
        self.sosChebyBP = np.zeros((1,1), dtype=float)
        self.sosChebyDC = np.zeros((1,1), dtype=float)
        self.notchB = np.zeros((1,), dtype=float)
        self.notchA = np.zeros((1,), dtype=float)

        self.fftThreshold = 0
        self.fftThresholdSDMultiplier = 0
        self.fftFoundFrequency = 0 
        self.fftFoundPower = 0 
        self.lockInThreshold = 0 
        self.lockInFrequency = 0 
        self.lockInThresholdFound = 0
        self.lockInStandardDevition = 0
        self.lockinThresholdSDMultiplier = 0

        self.fileName= None
        self.fileHandle = None
        self.fileStatus = None

        self.bufferReady = False
        self.continueLooping = False
        self.dataReady = False
        self.bufferOverrun = False
        self.threadRunning = False
        self.collectorRunning = False
        self.processorRunning = False
        self.reset = False
        self.rawData = False

    def printSizes(self):
        print("printSize timeBuffer shape",self.timeBuffer.shape)
        print("printSize amplitudeSine shape",self.amplitudeSine.shape)
        print("printSize amplitudeCosine shape",self.amplitudeCosine.shape)
        print("printSize amplitudeSN shape",self.amplitudeSN.shape)
        print("printSize bpFilteredSN shape",self.bpFilteredSN.shape)
        print("printSize sosChebyBP shape",self.sosChebyBP.shape)
        print("printSize sosChebyDC shape",self.sosChebyDC.shape)
        print("printSize notchB shape",self.notchB.shape)
        print("printSize notchA shape",self.notchA.shape)

    def printFlags(self):
        print(f"Flags: bufferReady:{self.bufferReady} continueLooping:{self.continueLooping} dataReady:{self.dataReady} bufferOverrun:{self.bufferOverrun} threadRunning:{self.threadRunning} collectorRunning:{self.collectorRunning} processorRunning:{self.processorRunning}")

    def getFileInfo (self, name):
        if (name == "fileHandle"):
            return self.fileHandle
        elif (name == "fileName"):
            return self.fileName
        elif (name == "fileStatus"):
            return self.fileStatus
        else:
            raise Exception(f"getFileInfo: Incorrect name {name} specified")

    def openFile (self, fileName):
        try:
            fh = open(fileName, 'w', newline='')
            self.fileName = fileName
            self.fileHandle = fh
            self.fileStatus = "open"
        except Exception as error:
            return error

        return None
    
    def writeFile (self, textLine):
        try:
            self.fileHandle.write(textLine) 
        except Exception as error:
            return error

        return None
    
    def closeFile (self):
        if (self.fileStatus == "open" and self.fileHandle != None):
            self.fileHandle.close()
            self.fileHandle = None
            self.fileName = None
            self.fileStatus = None

    def setAllDeviceInfo (self, currentDeviceIndex, currentDeviceName):
        self.currentDeviceIndex = currentDeviceIndex
        self.currentDeviceName = currentDeviceName
    def setDeviceInfo (self, name, value):
        if (name == "currentDeviceIndex"):
            self.currentDeviceIndex = value
        elif (name == "currentDeviceName"):
            self.currentDeviceName = value
        else:
            raise Exception(f"setDevice: Incorrect name {name} specified")

    def setAllFrequencies (self, frequency, frequencyDelta, sampleFrequency):
        self.sampleFrequency = sampleFrequency
        self.frequency = frequency
        self.frequencyDelta = frequencyDelta
    def setFrequency (self, name, value):
        if (name == "frequency"):
            self.frequency = value
        elif (name == "frequencyDelta"):
            self.frequencyDelta = value
        elif (name == "sampleFrequency"):
            self.sampleFrequency = value
        else:
            raise Exception(f"setFrequency: Incorrect name {name} specified")
  
    def updateBufferArray (self, name, value):
        if (name == "amplitudeSine"):
            self.amplitudeSine = value
        elif (name == "amplitudeCosine"):
            self.amplitudeCosine = value
        elif (name == "amplitudeSN"):
            self.amplitudeSN = value
        elif (name == "bpFilteredSN"):
            self.bpFilteredSN = value
        else:
            raise Exception(f"updateBufferArray: Incorrect name {name} specified")
           
    def setAllBufferInfo (self, sampleSize, samplesToProcess):
        self.sampleSize = sampleSize
        self.samplesToProcess = samplesToProcess
    def setBufferInfo (self, name, value):
        if (name == "sampleSize"):
            self.sampleSize = value
        elif (name == "samplesToProcess"):
            self.samplesToProcess = value
        else:
            raise Exception(f"setBuffersInfo: Incorrect name {name} specified")

    def zeroFFTInfo(self):
        self.fftThreshold = 0
        self.fftFoundFrequency =  0
        self.fftFoundPower = 0 
    def setAllFFTInfo(self, fftThreshold, fftThresholdSDMultiplier, fftFoundFrequency, fftFoundPower):
        self.fftThreshold = fftThreshold
        self.fftThresholdSDMultiplier = fftThresholdSDMultiplier
        self.fftFoundFrequency = fftFoundFrequency 
        self.fftFoundPower = fftFoundPower 
    def setFFTInfo(self, name, value):
        if (name == "fftThreshold"):
            self.fftThreshold = value
        elif (name == "fftThresholdSDMultiplier"):
            self.fftThresholdSDMultiplier = value
        elif (name == "fftFoundFrequency"):
            self.fftFoundFrequency = value
        elif (name == "fftFoundPower"):
            self.fftFoundPower = value
        else:
           raise Exception(f"setFFTInfo: Incorrect name {name} specified")

    def zeroLockinInfo(self):
        self.lockInThreshold = 0 
        self.lockInThresholdFound = 0
        self.lockInStandardDevition = 0
    def setAllLockinInfo(self, lockInThreshold, lockinThresholdSDMultiplier, lockInThresholdFound, lockInStandardDevition, lockInFrequency):
        self.lockInThreshold = lockInThreshold 
        self.lockinThresholdSDMultiplier = lockinThresholdSDMultiplier
        self.lockInThresholdFound = lockInThresholdFound
        self.lockInStandardDevition = lockInStandardDevition
        self.lockInFrequency = lockInFrequency
    def setLockinInfo(self, name, value):
        if (name == "lockInThreshold"):
            self.lockInThreshold = value
        elif (name == "lockinThresholdSDMultiplier"):
            self.lockinThresholdSDMultiplier = value
        elif (name == "lockInThresholdFound"):
            self.lockInThresholdFound = value
        elif (name == "lockInStandardDevition"): 
            self.lockInStandardDevition = value 
        elif (name == "lockInFrequency"): 
            self.lockInFrequency = value 
        else:
           raise Exception(f"setLockinInfo: Incorrect name {name} specified")

    def setAllFlags (self, bufferReady, continueLooping, dataReady, bufferOverrun, threadRunning, collectorRunning, processorRunning, reset):
        self.bufferReady = bufferReady
        self.continueLooping = continueLooping
        self.dataReady = dataReady
        self.bufferOverrun = bufferOverrun
        self.threadRunning = threadRunning
        self.collectorRunning = collectorRunning
        self.processorRunning = processorRunning
        self.reset = reset
    def setFlags (self, name, value):
        if (name == "bufferReady"):
            self.bufferReady = value
        elif (name == "continueLooping"):
            self.continueLooping = value
        elif (name == "dataReady"):
            self.dataReady = value
        elif (name == "bufferOverrun"):
            self.bufferOverrun = value
        elif (name == "threadRunning"):
            self.threadRunning = value
        elif (name == "collectorRunning"):
            self.collectorRunning = value
        elif (name == "processorRunning"):
            self.processorRunning = value        
        elif (name == "reset"):
            self.reset = value        
        elif (name == "rawData"):
            self.rawData = value        
        else:
            raise Exception(f"setFlags: Incorrect name {name} specified")

    def setFilterCoefficients(self, name, value):
        if (name == "sosChebyBP"):
            self.sosChebyBP = value
        elif (name == "sosChebyDC"):
            self.sosChebyDC = value
        elif (name == "notchB"):
            self.notchB = value
        elif (name == "notchA"):
            self.notchA = value
        else:
           raise Exception(f"setFilterCoefficients: Incorrect name {name} specified")
 
    def getAllFlags(self):
        return  self.bufferReady, self.continueLooping, self.dataReady, self.bufferOverrun, self.threadRunning, self.collectorRunning, self.processorRunning, self.reset, self.rawData
    def getFlags(self, name):
        if (name == "bufferReady"):
            return self.bufferReady 
        elif (name == "continueLooping"):
            return self.continueLooping
        elif (name == "dataReady"):
            return self.dataReady
        elif (name == "bufferOverrun"):
            return self.bufferOverrun
        elif (name == "threadRunning"):
            return self.threadRunning
        elif (name == "collectorRunning"):
            return self.collectorRunning
        elif (name == "processorRunning"):
            return self.processorRunning        
        elif (name == "reset"):
            return self.reset         
        elif (name == "rawData"):
            return self.rawData        
        else:
           raise Exception(f"getFlags: Incorrect name {name} specified")

    def getAllFrequencyInfo(self):
        return self.frequency, self.sampleSize
    def getFrequency (self, name):
        if (name == "frequency"):
            return self.frequency 
        elif (name == "frequencyDelta"):
            return self.frequencyDelta 
        elif (name == "sampleFrequency"):
            return self.sampleFrequency 
        else:
            raise Exception(f"getFrequency: Incorrect name {name} specified")

    def getAllDeviceInfo(self):
        return self.currentDeviceIndex, self.currentDeviceName
    def getDeviceInfo(self, name):
        if (name == "currentDeviceIndex"):
            return self.currentDeviceIndex 
        elif (name == "currentDeviceName"):
            return self.currentDeviceName 
        else:
            raise Exception(f"getDeviceInfo: Incorrect name {name} specified")

    def getBufferArray (self, name):
       if (name == "amplitudeSine"):
           return self.amplitudeSine
       elif (name == "amplitudeCosine"):
           return self.amplitudeCosine 
       elif (name == "amplitudeSN"):
           return self.amplitudeSN
       elif (name == "bpFilteredSN"):
           return self.bpFilteredSN 
       else:
           raise Exception(f"getBufferArray: Incorrect name {name} specified")

    def getAllBufferInfo(self):
        return self.sampleSize, self.samplesToProcess
    def getBufferInfo(self, name):
       if (name == "sampleSize"):
           return self.sampleSize
       elif (name == "samplesToProcess"):
           return self.samplesToProcess 
       else:
           raise Exception(f"getBufferInfo: Incorrect name {name} specified")

    def getAllFFTInfo(self):
        return self.fftThreshold, self.fftThresholdSDMultiplier, self.fftFoundFrequency, self.fftFoundPower
    def getFFTInfo(self, name):
       if (name == "fftThreshold"):
           return self.fftThreshold
       elif (name == "fftThresholdSDMultiplier"):
           return self.fftThresholdSDMultiplier 
       elif (name == "fftFoundFrequency"):
           return self.fftFoundFrequency 
       elif (name == "fftFoundPower"):
           return self.fftFoundPower 
       else:
           raise Exception(f"getFFTInfo: Incorrect name {name} specified")
 
    def getAllLockinInfo(self):
        return self.lockInThreshold, self.lockinThresholdSDMultiplier, self.lockInThresholdFound, self.lockInStandardDevition, self.lockInFrequency 
    def getLockinInfo(self, name):
       if (name == "lockInThreshold"):
           return self.lockInThreshold
       elif (name == "lockinThresholdSDMultiplier"):
           return self.lockinThresholdSDMultiplier 
       elif (name == "lockInThresholdFound"):
           return self.lockInThresholdFound 
       elif (name == "lockInStandardDevition"):
           return self.lockInStandardDevition 
       elif (name == "lockInFrequency"):
           return self.lockInFrequency 
       else:
           raise Exception(f"getLockinInfo: Incorrect name {name} specified")

    def getFilterCoefficients(self, name):
        if (name == "sosChebyBP"):
            return self.sosChebyBP 
        elif (name == "sosChebyDC"):
            return self.sosChebyDC 
        elif (name == "notchB"):
            return self.notchB 
        elif (name == "notchA"):
            return self.notchA 
        else:
           raise Exception(f"getFilterCoefficients: Incorrect name {name} specified")
 
def mainWindow():
    global statusLabel, entButton, exitButton, fileButton, stopButton, testButton, supportedFrequencies, frequencyComboBox, resetButton
    global filteredSoundInputDeviceNames, deviceComboBox, fftSDMultiplier, lockinSDMultiplier, rawEnableButton, rawEnable
    global supportedSampleRates, sampleRateComboBox, frequencyDeltaComboBox, sampleRates, sampleBitSizes

    global threadId, threadEvent, canvas, toolbar, fig, ax, ax2, plotx, ploty, plotz, yTrendLine, zTrendLine

    global sdi, manager



    BaseManager.register('SignalDection', SignalDection)
    manager = BaseManager()
    manager.start()
    sampleSize = 4096
    samplesToProcess = sampleSize
    sdi = manager.SignalDection(72, None, 44100, 1300, 60, sampleSize, samplesToProcess)    
    sdi.setAllFFTInfo(0,0,0,0)
    sdi.setAllLockinInfo(0,0,0,0,0)

    ###############  Status Windows
    statusLabel = tb.Label(statusFrame, text="Ok", bootstyle="primary", font=("Times", 10, "bold italic"), width=70, relief=RAISED)
    statusLabel.grid(row=0, column=0, padx=10, pady=2, sticky="W")

    ###############  Control Buttons Windows
    entButton = tb.Button(controlRow1Frame, text="Start", width=6, bootstyle="success-outline", 
                          command=lambda:entButtonHandler(sd, sdi))
    entButton.configure(state=tb.NORMAL)
    entButton.grid(row=0, column=0, padx=10, pady=2, sticky="W")
    entButton.grid_columnconfigure(0, weight=0)

    stopButton = tb.Button(controlRow1Frame, text="Stop", width=5, bootstyle="warning-outline", command=lambda:stopButtonHandler(sdi))
    stopButton.configure(state=tb.DISABLED)
    stopButton.grid(row=0, column=1, padx=10, pady=2, sticky="W")
    stopButton.grid_columnconfigure(1, weight=0)

    resetButton = tb.Button(controlRow1Frame, text="Reset Plot", width=11, bootstyle="warning-outline", command=lambda:resetPlot(sdi))
    resetButton.configure(state=tb.DISABLED)
    resetButton.grid(row=0, column=2, padx=10, pady=2, sticky="W")
    resetButton.grid_columnconfigure(2, weight=0)

    fileButton = tb.Button(controlRow1Frame, text="File Open", width=10, bootstyle="success-outline", command=lambda:fileButtonHandler(sdi))
    fileButton.configure(state=tb.NORMAL)
    fileButton.grid(row=0, column=3, padx=10, pady=2, sticky="W")
    fileButton.grid_columnconfigure(3, weight=0)

    testButton = tb.Button(controlRow1Frame, text="Test", width=5, bootstyle="success-outline", command=lambda:testButtonHandler(sdi))
    testButton.configure(state=tb.NORMAL)
    testButton.grid(row=0, column=4, padx=10, pady=2, sticky="W")
    testButton.grid_columnconfigure(4, weight=0)

    exitButton = tb.Button(controlRow1Frame, text="Exit", width=5, bootstyle="success-outline", command=lambda:exitButtonHandler(sdi))
    exitButton.configure(state=tb.NORMAL)
    exitButton.grid(row=0, column=5, padx=10, pady=2, sticky="W")
    stopButton.grid_columnconfigure(5, weight=0)

    ###############  Configuration Controls Windows

    # soundDeviceList = sd.query_devices()
    # filteredSoundInputDeviceNames = soundDeviceList
    filteredSoundInputDeviceNames = []
    soundInputDeviceNames = []
    apiNames= []
    sampleRates = [44100, 48000, 96000]
    deltaValues = [60, 100, 200]
    sampleBitSizes = [16, 24]
    multiplerValues = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    supportedFrequencies = [CONST.FREQ_700, CONST.FREQ_900, CONST.FREQ_1100,
                CONST.FREQ_1300, CONST.FREQ_1500, CONST.FREQ_1700, CONST.FREQ_1900, CONST.FREQ_2100,
                CONST.FREQ_2300, CONST.FREQ_2500, CONST.FREQ_2700, CONST.FREQ_2900]


    # for id in range(len(soundDeviceList)):
    #     if (soundDeviceList[id]['max_input_channels'] > 0 and soundDeviceList[id]['default_samplerate'] > 0):
    #         api = sd.query_hostapis(index=soundDeviceList[id]['hostapi'])['name']
    #         soundInputDeviceNames.append(f"{soundDeviceList[id]['name']} ({api})" )
    #         filteredSoundInputDeviceNames.append(soundDeviceList[id])
    #         apiNames.append(api)    

    soundInputDeviceNames, apiNames, filteredSoundInputDeviceNames = generateSoundDevices()

     
    deviceComboBox = tb.Combobox(controlRow2Frame, bootstyle="sucess", width=78, font=("Times", 8), values=soundInputDeviceNames)
    deviceComboBox.grid(row=0, column=0, padx=10, pady=2, columnspan=6, sticky="NSEW")
    #deviceComboBox.grid_columnconfigure(0, weight=1)
    deviceComboBox.current(3)
    deviceComboBox.configure(state=tk.NORMAL)
    deviceComboBox.bind("<<ComboboxSelected>>", lambda evnt: deviceHandler(evnt, sdi))
    sdi.setAllDeviceInfo(int(filteredSoundInputDeviceNames[deviceComboBox.current()]['index']), deviceComboBox.get())

    sampleRateLabel = tb.Label(controlRow3Frame, text="Sample Rate", bootstyle="light", font=("Times", 8))
    sampleRateLabel.grid(row=0, column=0, padx=10, pady=2, sticky="W")
    supportedSampleRates = getSupportedRates(sampleRates, int(filteredSoundInputDeviceNames[deviceComboBox.current()]['index']))
    sampleRateComboBox = tb.Combobox(controlRow3Frame, bootstyle="sucess", width=6, values=supportedSampleRates)
    sampleRateComboBox.grid(row=1, column=0, padx=10, pady=2, sticky="W")
    sampleRateComboBox.grid_columnconfigure(0, weight=0)
    sampleRateComboBox.set(44100)
    sampleRateComboBox.configure(state=tk.NORMAL)
    sampleRateComboBox.bind("<<ComboboxSelected>>", lambda evnt: deviceHandler(evnt, sdi))
    sdi.setFrequency("sampleFrequency", int(supportedSampleRates[sampleRateComboBox.current()]))

    fftSDMultLabel = tb.Label(controlRow3Frame, text="FFT SD Mult", bootstyle="light", font=("Times", 8))
    fftSDMultLabel.grid(row=0, column=1, padx=10, pady=2, sticky="W")
    fftSDMultiplier = tb.Spinbox(controlRow3Frame, bootstyle="Delta", width=4, from_=0, to=10, wrap=True, command=lambda: updateDataHandler("", sdi))
    fftSDMultiplier.grid(row=1, column=1, padx=10, pady=2, sticky="W")
    fftSDMultiplier.grid_columnconfigure(1, weight=0)
    fftSDMultiplier.set(0)
    fftSDMultiplier.configure(state=tk.NORMAL)
    sdi.setFFTInfo ("fftThresholdSDMultiplier", int(fftSDMultiplier.get()))

    LockinSDMultLabel = tb.Label(controlRow3Frame, text="LckIn SD Mult", bootstyle="light", font=("Times", 8))
    LockinSDMultLabel.grid(row=0, column=2, padx=10, pady=2, sticky="W")
    lockinSDMultiplier = tb.Spinbox(controlRow3Frame, bootstyle="Delta", width=4, from_=0, to=10, wrap=True, command=lambda: updateDataHandler("", sdi))
    lockinSDMultiplier.grid(row=1, column=2, padx=10, pady=2, sticky="W")
    lockinSDMultiplier.grid_columnconfigure(2, weight=0)

    lockinSDMultiplier.set(0)
    lockinSDMultiplier.configure(state=tk.NORMAL)
    sdi.setLockinInfo ("lockinThresholdSDMultiplier", int(lockinSDMultiplier.get()))

    frequencyLabel = tb.Label(controlRow3Frame, text="Frequency", bootstyle="light", font=("Times", 8))
    frequencyLabel.grid(row=0, column=3, padx=10, pady=2, sticky="W")
    frequencyComboBox = tb.Combobox(controlRow3Frame, bootstyle="sucess", width=5, values=supportedFrequencies)
    frequencyComboBox.grid(row=1, column=3, padx=10, pady=2, sticky="W")
    frequencyComboBox.grid_columnconfigure(3, weight=0)
    frequencyComboBox.bind("<<ComboboxSelected>>", lambda evnt: deviceHandler(evnt, sdi))
    # frequencyComboBox.current(6)
    frequencyComboBox.set(CONST.FREQ_1300)
    frequencyComboBox.configure(state=tk.NORMAL)
    sdi.setFrequency("frequency", int(frequencyComboBox.get()))

    frequencyDeltaLabel = tb.Label(controlRow3Frame, text="Freq. Delta", bootstyle="light", font=("Times", 8))
    frequencyDeltaLabel.grid(row=0, column=4, padx=10, pady=2, sticky="W")
    frequencyDeltaComboBox = tb.Combobox(controlRow3Frame, bootstyle="Delta", width=6, values=deltaValues)
    frequencyDeltaComboBox.grid(row=1, column=4, padx=10, pady=2, sticky="W")
    frequencyDeltaComboBox.grid_columnconfigure(4, weight=0)
    frequencyDeltaComboBox.current(0)
    frequencyDeltaComboBox.configure(state=tk.NORMAL)
    frequencyDeltaComboBox.bind("<<ComboboxSelected>>", lambda evnt: updateDataHandler(evnt, sdi))
    sdi.setFrequency ("frequencyDelta", int(frequencyDeltaComboBox.get()))


    rawEnable = tb.BooleanVar()
    rawEnable.set(False)
    rawEnableButton = tb.Checkbutton(controlRow3Frame, text='Raw Mode', style='Roundtoggle.Toolbutton', 
                                       onvalue=True, offvalue=False, variable=rawEnable, command=lambda: updateDataHandler(None, sdi))
    rawEnableButton.configure(state=tk.NORMAL) 
    rawEnableButton.grid(row=1, column=5, padx=10, pady=2, sticky="W")
    rawEnableButton.grid_columnconfigure(5, weight=0)



###############  Plot Windows 
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    #ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax.autoscale(enable=True, axis='y')
    #ax2.autoscale(enable=True, axis='y')
    plotx = []
    ploty = []
    plotz = []
    yTrendLine = [] 
    zTrendLine = []

    canvas = FigureCanvasTkAgg(fig, master=chartFrame)  # A tk.DrawingArea.
    canvas.get_tk_widget().pack()

    # # pack_toolbar=False will make it easier to use a layout manager later on.
    # toolbar = NavigationToolbar2Tk(canvas, chartFrame, pack_toolbar=True)
    # toolbar.update()

    # canvas.mpl_connect(
    #     "key_press_event", lambda event: print(f"you pressed {event.key}"))
    # canvas.mpl_connect("key_press_event", key_press_handler)

############### Setup for run with defaults
    sd.default.samplerate = sdi.getFrequency ("sampleFrequency")
    sd.default.device = sdi.getDeviceInfo("currentDeviceIndex")
    sd.default.channels = 1
    sd.default.dtype = 'int16'

    sdi.setAllFlags (False, False, False, False, False, False, False, False)

def resetPlot(sdi):
    global canvas, fig, ax, plotx, ploty, plotz, yTrendLine, zTrendLine, resetPlotData

    sdi.setFlags("reset", True)

def updatePlot(timepoint, lockinpoint, std, fftpoint, fftFreq, lockinFreq):
    global canvas, fig, ax, plotx, ploty, plotz, yTrendLine, zTrendLine, sdi, t2
    if (sdi.getFlags("reset")):
        plotx = []
        ploty = []
        plotz = []
        yTrendLine = []
        zTrendLine = []
        ax.clear()
        sdi.setFlags("reset", False)

    plotx.append(timepoint)
    ploty.append(lockinpoint)
    plotz.append(fftpoint)
    if (len(plotx) > 4):
        yAvg = np.mean(ploty[-4:])
        zAvg = np.mean(plotz[-4:])
    else: 
        yAvg = np.mean(ploty)
        zAvg = np.mean(plotz)
    yTrendLine.append(yAvg)
    zTrendLine.append(zAvg)

    if (sdi.getFileInfo("fileStatus") == "open"):
        # timeText = str(datetime.timedelta(seconds=timepoint))
        timeText = str(datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S.%f'))
        textLine = f"{timeText}, {lockinFreq}, {lockinpoint}, {std}, {fftFreq}, {fftpoint} \r\n"
        errorStatus = sdi.writeFile(textLine)
        if (errorStatus != None):
            print (errorStatus)

    ax.scatter(plotx, ploty, c ="cyan", linewidths = 1, marker ="o", edgecolor ="black", s = 10, alpha=0.5)
    #ax2.scatter(plotx, plotz, c ="green", linewidths = 1, marker ="8", edgecolor ="magenta", s = 10, alpha=0.5)
    ax.scatter(plotx, plotz, c ="green", linewidths = 1, marker ="8", edgecolor ="magenta", s = 10, alpha=0.5)
    ax.legend(['Lockin', 'FFT'])
    #ax2.plot(plotx, yTrendLine, color='b', linestyle='-') 
    ax.plot(plotx, yTrendLine, color='cyan', linestyle='-')
    ax.plot(plotx, zTrendLine, color='green', linestyle='-') 
    #ax2.legend(['FFT'])
    
    if (len(ax.texts) > 0):
        for lin in ax.texts:
            lin.remove()

    if (fftFreq > 0):
        textVal = ax.text(.85, .05, f"FFT Freq: {int(fftFreq)}", fontsize=8, horizontalalignment='center', 
            verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='green', alpha=0.5))
        Artist.set_visible(textVal, True)

    if (lockinFreq > 0 ):
        textVal2 = ax.text(.15, .05, f"Lck Freq: {int(lockinFreq)}", fontsize=8, horizontalalignment='center', 
            verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='cyan', alpha=0.5))
        Artist.set_visible(textVal2, True)

    canvas.draw()

    block = False

def getSupportedRates(sampleRates, currentDeviceIndex):
    supportedSampleRates = []
    for rate in sampleRates:
        try:
            sd.default.samplerate = rate
            sd.default.device = currentDeviceIndex
            sd.default.channels = 1
            sd.check_input_settings(device=currentDeviceIndex, samplerate=rate)
            checkData = sd.rec(frames=32, samplerate=rate)
            supportedSampleRates.append(rate)
        except Exception as error:
            pass

        if (len(supportedSampleRates) == 0):
            supportedSampleRates = ["None"]
    return supportedSampleRates

def deviceHandler(e, sdi):
    global statusLabel, entButton, exitButton, fileButton, stopButton, testButton, frequencyComboBox, filteredSoundInputDeviceNames, deviceComboBox
    global fftSDMultiplier, lockinSDMultiplier
    global sampleRateComboBox, frequencyDeltaComboBox, sampleRates, supportedSampleRates

    deviceIndex = filteredSoundInputDeviceNames[deviceComboBox.current()]["index"]
    arrayIndex = deviceComboBox.current()

    if (sdi.getDeviceInfo ("currentDeviceIndex") != deviceIndex ):    
        sdi.setDeviceInfo("currentDeviceName", deviceComboBox.get())   
        sdi.setDeviceInfo("currentDeviceIndex", int(filteredSoundInputDeviceNames[deviceComboBox.current()]['index']))  
        supportedSampleRates = getSupportedRates(sampleRates, deviceIndex)
        index = supportedSampleRates.index(filteredSoundInputDeviceNames[arrayIndex]['default_samplerate'])
        sampleRateComboBox.config(values=supportedSampleRates)
        sampleRateComboBox.current(index)
    else: 
        statusLabel.config(text="Ok", bootstyle="primary")

    sdi.setDeviceInfo("currentDeviceName", deviceComboBox.get())   
    sdi.setDeviceInfo("currentDeviceIndex", int(filteredSoundInputDeviceNames[deviceComboBox.current()]['index']))
    sdi.setFrequency("sampleFrequency", int(supportedSampleRates[sampleRateComboBox.current()]))
    sdi.setFrequency("frequency", int(frequencyComboBox.get()))
    sdi.setFrequency("frequencyDelta", int(frequencyDeltaComboBox.get()))
    sdi.setFFTInfo ("fftThresholdSDMultiplier", int(fftSDMultiplier.get()))
    sdi.setLockinInfo ("lockinThresholdSDMultiplier", int(lockinSDMultiplier.get()))

    sd.default.samplerate = sdi.getFrequency("sampleFrequency")
    sd.default.device = sdi.getDeviceInfo("currentDeviceIndex")
    sd.default.channels = 1
    sd.default.dtype = 'int32'

def updateDataHandler(e, sdi):
    global sampleRateComboBox, frequencyDeltaComboBox, sampleRates, supportedSampleRates, rawEnableButton, rawEnable

    sdi.setFrequency("frequencyDelta", int(frequencyDeltaComboBox.get()))
    sdi.setFFTInfo ("fftThresholdSDMultiplier", int(fftSDMultiplier.get()))
    sdi.setLockinInfo ("lockinThresholdSDMultiplier", int(lockinSDMultiplier.get()))
    sdi.setFlags ("rawData", rawEnable.get())

def testButtonHandler(sdi):

    # sd.default.samplerate = sdi.getFrequency("sampleFrequency")
    # sd.default.device = sdi.getDeviceInfo("currentDeviceIndex")
    # sd.default.channels = 1
    # sd.default.dtype = 'int32'
    # sampleSize = sdi.getBufferInfo("sampleSize")
    # sampleFrequency = sdi.getFrequency("sampleFrequency")
    # sdi.zeroFFTInfo()
    # sdi.zeroLockinInfo()


    # sdi.setFrequency("sampleFrequency", 48000)
    # sdi.setFrequency("frequency", 1300)

    # sampleSize = sdi.getBufferInfo("sampleSize")
    # freq = sdi.getFrequency("frequency")
    # rate = sdi.getFrequency("sampleFrequency")
    # endTime = sampleSize / rate 
    # theta = 0

    # print (F"{freq}, {rate}, {sampleSize}, {endTime}")
    # timeBuffer = np.arange(0, endTime, 1/rate)

    # print (F"Buffer inital len: {len(timeBuffer)} for {freq} {rate} {sampleSize}")

    # # Amplitude of the sine wave is sine of a variable like time

    # sinBuff = np.sin(2 * np.pi * freq * timeBuffer + theta)
    # cosBuff = np.cos(2 * np.pi * freq * timeBuffer + theta)
    # plot.clf()
    # plot.plot(timeBuffer[0:512], sinBuff[0:512], 'v-b')
    # plot.plot(timeBuffer[0:512], cosBuff[0:512], '^-b')

    # sdi.setFrequency("sampleFrequency", 16000)
    # sdi.setFrequency("frequency", 1300)

    # sampleSize = sdi.getBufferInfo("sampleSize")
    # freq = sdi.getFrequency("frequency")
    # rate = sdi.getFrequency("sampleFrequency")
    # endTime = sampleSize / rate 
    # theta = 0

    # print (F"{freq}, {rate}, {sampleSize}, {endTime}")
    # timeBuffer = np.arange(0, endTime, 1/rate)
    # print (F"Buffer inital len: {len(timeBuffer)} for {freq} {rate} {sampleSize}")

    # # Amplitude of the sine wave is sine of a variable like time

    # sinBuff2 = np.sin(2 * np.pi * freq * timeBuffer + theta)
    # cosBuff2 = np.cos(2 * np.pi * freq * timeBuffer + theta)
    # plot.plot(timeBuffer[0:512], sinBuff2[0:512], '<-m')
    # plot.plot(timeBuffer[0:512], cosBuff2[0:512], '>-m')

    # plot.title('testButtonHandler Reference Signals')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()

    # return

    # t1 = timer.perf_counter()
    # sdi.setFlags("threadRunning", True)
    # t2 = timer.perf_counter()
    # print ("setFlag: ", (t2-t1)*1000000)

    # t1 = timer.perf_counter()
    # sdi.getFlags("threadRunning")
    # t2 = timer.perf_counter()
    # print ("getFlag: ", (t2-t1)*1000000)


    # buff = np.arange (0, sampleSize, 1)
    # t1 = timer.perf_counter()
    # sdi.updateBufferArray("amplitudeSN", buff)
    # t2 = timer.perf_counter()
    # print ("Update buffer: ", (t2-t1)*1000000)

    # t1 = timer.perf_counter()
    # buff = sdi.getBufferArray("amplitudeSN")
    # t2 = timer.perf_counter()
    # print ("Read buffer: ", (t2-t1)*1000000)

    # if (sdi.getFlags("threadRunning")):
    #     sdi.setAllFlags (False, False, False, False, False, False, False, False)
    #     return


    print ("SD Defaults")
    print ("  Device Index: ", sd.default.device)
    print ("  Sample Rate: ", sd.default.samplerate)
    print ("  Channels: ", sd.default.channels)
    print ("  Data Type: ", sd.default.dtype)

    # generateData(sdi)

    # amplitudeSine = sdi.getBufferArray("amplitudeSine")
    # amplitudeCosine = sdi.getBufferArray("amplitudeCosine")
    # plot.clf()
    # plot.plot(range(len(amplitudeSine)), amplitudeSine, 'o-r')
    # plot.plot(range(len(amplitudeCosine)), amplitudeCosine, '8-r')
    # plot.title('testButtonHandler Reference Signals')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()

    # notchFilterSetup(sdi)
    # chebyFilterSetup(sdi)

    # sampleSize = sdi.getBufferInfo("sampleSize")
    # sampleFrequency = sdi.getFrequency("sampleFrequency")
    # duration = 1.1 * sampleSize / sampleFrequency

    # sdi.updateBufferArray("amplitudeSN", sdi.getBufferArray("amplitudeSine"))

    # t1 = timer.perf_counter()
    # myrecording = sd.rec(int(duration * sampleFrequency), blocking=True)
    # buff = myrecording[:sampleSize,0]
    # sdi.updateBufferArray("amplitudeSN", buff)
    # t2 = timer.perf_counter()
    # print ("time to acquire: ", t2-t1)

    # amplitudeSN = sdi.getBufferArray("amplitudeSN")
    # plot.clf()
    # plot.plot(range(len(amplitudeSN)), amplitudeSN, 'o-r')
    # plot.title('testButtonHandler Signal')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()
    # t1 = timer.perf_counter()
    # filterData (sdi)

    # bpFilteredSN = sdi.getBufferArray("bpFilteredSN")
    # plot.clf()
    # plot.plot(range(len(bpFilteredSN)), bpFilteredSN, 'o-r')
    # plot.title('testButtonHandler Filtered Signal')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()

    # getNoiseFloor (sdi)
    # lockInAmplifier (sdi)

    # plot.clf()
    # plot.plot(range(len(lockInAmlitude)), lockInAmlitude, 'o-r')
    # thres = lockInAmlitude
    # thres.fill(lockInThreshold)
    # plot.plot(range(len(lockInAmlitude)), thres, "k") 
    # plot.title('testButtonHandler Lockin Amp')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.legend(['Lockin', 'Thresh'])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show() 

    # performFFTonData (sdi)

    # t2 = timer.perf_counter()
    # print ("time to process: ", t2-t1)

    # lockInThresholdFound = sdi.getLockinInfo("lockInThresholdFound")
    # fftFoundFrequency = sdi.getFFTInfo("fftFoundFrequency")
    # fftFoundPower = sdi.getFFTInfo("fftFoundPower")
    # print (f"Lockin: {lockInThresholdFound}, FFT: {fftFoundFrequency} {fftFoundPower}")
 
def entButtonHandler(sd, sdi):
    global statusLabel, entButton, exitButton, testButton, stopButton, frequencyComboBox, deviceComboBox
    global sampleRateComboBox, fftSDMultiplier, lockinSDMultiplier  

    global threadId, threadEvent, ax, plotx, ploty, plotz, yTrendLine, zTrendLine

    try:
        sd.check_input_settings()
    except:
        statusLabel.config(text="Audio Device Check Failed", bootstyle="danger")
        return

    threadEvent = Event()
    threadEvent.clear()

    generateData(sdi)
    notchFilterSetup(sdi)
    chebyFilterSetup(sdi)

    sdi.zeroFFTInfo()
    sdi.zeroLockinInfo()
    sdi.setAllFlags (False, True, False, False, False, False, False, False)  # continueLooping True

    threadId = threading.Thread(target=mainThread, args=(statusLabel, threadEvent, sdi))
    threadId.start()

    entButton.configure(state=tk.DISABLED)
    stopButton.configure(state=tk.NORMAL)
    fileButton.configure(state=tb.DISABLED)
    exitButton.configure(state=tk.DISABLED)
    testButton.configure(state=tk.DISABLED)
    resetButton.configure(state=tk.NORMAL)
    frequencyComboBox.configure(state=tk.DISABLED)
    deviceComboBox.configure(state=tk.DISABLED)
    sampleRateComboBox.configure(state=tk.DISABLED)
    frequencyDeltaComboBox.configure(state=tk.NORMAL)
    fftSDMultiplier.configure(state=tk.NORMAL)
    lockinSDMultiplier.configure(state=tk.NORMAL)
    rawEnableButton.configure(state=tk.DISABLED)


    plotx = []
    ploty = []
    plotz = []
    yTrendLine = []
    zTrendLine = []
    ax.clear()
    #ax2.clear()

def fileButtonHandler(sdi):
    global statusLabel, entButton, exitButton, fileButton, stopButton, testButton, frequencyComboBox, deviceComboBox
    global sampleRateComboBox, frequencyDeltaComboBox, fftSDMultiplier, lockinSDMultiplier

    global threadId, threadEvent
    # global bufferProcessorProcessId, soundRecevierProcessId

    if (sdi.getFileInfo("fileStatus") == "open"):
        if (messagebox.askokcancel(title="Confirm Close", message="File Close Confirmation")):
            sdi.closeFile()
            fileButton.configure(text="File Open")
    else:
        my_filetypes = [("Excel CSV files", ".csv .txt")]
        fileName = ''
        fileName = filedialog.asksaveasfilename(parent=win, initialdir=os.getcwd(
        ), title="Please select a file name for saving:", filetypes=my_filetypes)

        if len(fileName) == 0:
            pass
        else:
            fileButton.configure(text="File Close")
            errorStatus = sdi.openFile(fileName)
            if (errorStatus != None):
                messagebox.showerror (title="File Open Error", message=f"Could not open file {fileName}")
                return
            textLine = f"{banner}\r\n\r\ntime, lockinFreq, lockInAmplitude, lockInStdDev, fftFrequency, fftPower\r\n"
            errorStatus = sdi.writeFile (textLine)
            if (errorStatus != None):
                messagebox.showerror (title="File Write Error", message=f"Could write to file {fileName}. Closing file")
                sdi.closeFile()
                return
            fileButton.configure(text="File Close")

def stopButtonHandler(sdi):
    global statusLabel, entButton, exitButton, fileButton, stopButton, testButton, frequencyComboBox, deviceComboBox
    global sampleRateComboBox, frequencyDeltaComboBox, fftSDMultiplier, lockinSDMultiplier, filteredSoundInputDeviceNames

    global threadId, threadEvent
    # global soundRecevierProcessId, bufferProcessorProcessId

    if (sdi.getFlags("continueLooping") or sdi.getFlags("threadRunning") or 
        sdi.getFlags("processorRunning") or sdi.getFlags("collectorRunning")):
        sdi.setFlags("continueLooping", False)
        threadEvent.set()
        timer.sleep(1.5)

    sdi.setAllFlags(False, False, False, False, False, False, False, False)

    entButton.configure(state=tk.NORMAL)
    stopButton.configure(state=tk.DISABLED)
    fileButton.configure(state=tb.NORMAL)
    exitButton.configure(state=tk.NORMAL)
    testButton.configure(state=tk.NORMAL)
    resetButton.configure(state=tk.DISABLED)
    frequencyComboBox.configure(state=tk.NORMAL)
    deviceComboBox.configure(state=tk.NORMAL)
    sampleRateComboBox.configure(state=tk.NORMAL)
    frequencyDeltaComboBox.configure(state=tk.NORMAL)
    fftSDMultiplier.configure(state=tk.NORMAL)
    lockinSDMultiplier.configure(state=tk.NORMAL)
    rawEnableButton.configure(state=tk.NORMAL)

    soundInputDeviceNames, apiNames, filteredSoundInputDeviceNames = generateSoundDevices()
    deviceComboBox.configure(values=soundInputDeviceNames)

    if (sdi.getFileInfo("fileStatus") == "open"):
        if (not messagebox.askyesno(title="Keep File Open", message="Do you want to keep the file open so that you can continue writing to it?")):
            sdi.closeFile()
            fileButton.configure(text="File Open")        

def generateSoundDevices():

    soundDeviceList = sd.query_devices()
    filteredSoundInputDeviceNames = soundDeviceList
    filteredSoundInputDeviceNames = []
    soundInputDeviceNames = []
    apiNames= []

    for id in range(len(soundDeviceList)):
        if (soundDeviceList[id]['max_input_channels'] > 0 and soundDeviceList[id]['default_samplerate'] > 0):
            api = sd.query_hostapis(index=soundDeviceList[id]['hostapi'])['name']
            soundInputDeviceNames.append(f"{soundDeviceList[id]['name']} ({api})" )
            filteredSoundInputDeviceNames.append(soundDeviceList[id])
            apiNames.append(api) 

    return soundInputDeviceNames, apiNames, filteredSoundInputDeviceNames

def closeWindow():
    global sdi, manager
    if (sdi.getFlags("continueLooping") or sdi.getFlags("threadRunning") or 
        sdi.getFlags("processorRunning") or sdi.getFlags("collectorRunning")):
        messagebox.showwarning(title="Warning", message="Background threads are still running. You must stop them first")
        return
    else:
        okToExit = messagebox.askokcancel(title="Exit", message="Are you sure you want to exit?")
        if (okToExit):
            win.quit()
            sys.exit()

def exitButtonHandler(sdi):

    okToExit = messagebox.askokcancel(title="Exit", message="Are you sure you want to exit?")
    if (okToExit):
        if (sdi.getFlags("continueLooping") or sdi.getFlags("collectorRunning") 
            or sdi.getFlags("processorRunning") or sdi.getFlags("threadRunning")):
            sdi.setAllFlags(False, False, False, False, False, False, False, False)
            threadEvent.set()
            timer.sleep(1.5)

        win.quit()
        sys.exit()

def mainThread(statusLabel, threadEvent, sdi):

    sd.default.samplerate = sdi.getFrequency ("sampleFrequency")
    sd.default.device = sdi.getDeviceInfo("currentDeviceIndex")
    sd.default.channels = 1
    sd.default.dtype = 'int32'  

    sampleSize = sdi.getBufferInfo("sampleSize")
    samplesToProcess = sdi.getBufferInfo("samplesToProcess")
    sampleFrequency = sdi.getFrequency("sampleFrequency")
    sdi.setAllFlags (False, True, False, False, False, False, False, False)

    # Input device for ICOM Radio: 72 Microphone (C-Media USB Audio Device), Windows WDM-KS (1 in, 0 out)
    '''
    Sample Time 48000, Sample period: 20.93 uS
    Signal frequency 1300, Period: 769.2uS
    For each signal period, there are 36.92 samples taken
    36 samples are for 1 wave length
    100 waves, requires 3692
    Sample  Freq    SamPer  SigPer  Sam/Wav Buffer/100Wav 
    48000	1300	20.8	769.2	36.9	3692
    24000	1300	41.7	769.2	18.5	1846
    12000	1300	83.3	769.2	9.2	    923
    6000	1300	166.7	769.2	4.6	    462
    '''

    sdi.setFlags("threadRunning", True)
    sdi.setFlags("continueLooping", True)

    statusLabel.config(text="Processing", bootstyle="sucess")
    
    now = datetime.datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    with sd.InputStream(samplerate=sampleFrequency, blocksize=sampleSize, dtype = 'int32',  
                        callback=lambda indata, frames, time, status: soundCallback(indata, frames, time, status, sdi)):
        while (sdi.getFlags("continueLooping")):
            if (sdi.getFlags("dataReady")):
                filterData (sdi)
                getNoiseFloor (sdi)
                sdi.setFlags("dataReady", False)
                lockInAmplifier (sdi)
                performFFTonData (sdi)
                    
                lockInThresholdFound = sdi.getLockinInfo("lockInThresholdFound")
                lockInStandardDevition = sdi.getLockinInfo("lockInStandardDevition")
                fftFoundFrequency = sdi.getFFTInfo("fftFoundFrequency")
                lockinFoundFrequency = sdi.getLockinInfo("lockInFrequency")
                fftFoundPower = sdi.getFFTInfo("fftFoundPower")
                sdi.setFlags("dataReady", False)

                # print (f"mainThread: Lockin: {lockInThresholdFound} FFT Freq: {fftFoundFrequency} FFT Power: {fftFoundPower}")

                stat = ""
                now = datetime.datetime.now()
                seconds = (now - midnight).seconds

                stat = f"Lockin: {round(lockinFoundFrequency,1)}:{round(lockInThresholdFound,3)} FFT: {round(fftFoundFrequency,1)}:{round(fftFoundPower,3)}"
                statusLabel.config(text=stat, bootstyle="success")
                updatePlot(seconds, lockInThresholdFound, lockInStandardDevition, fftFoundPower, fftFoundFrequency, lockinFoundFrequency)  #(time, lockin, fft, freq)

            if (sdi.getFlags("bufferOverrun")):
                statusLabel.config(text="Overflow", bootstyle="danger")
                sdi.setFlags("bufferOverrun", False)
    
            if (threadEvent.isSet()):
                sdi.setFlags("continueLooping", False)
                break

    sdi.setFlags("continueLooping", False)
    statusLabel.config(text="Process Stopped", bootstyle="primary")
    sdi.setAllFlags (False, False, False, False, False, False, False, False)

def soundCallback(indata, frames, time, status, sdi):
    sampleSize = sdi.getBufferInfo("sampleSize")
    sdi.setFlags("collectorRunning", True)

    if (not bool(sdi.getFlags("continueLooping"))):
        sdi.setFlags("collectorRunning", False)
        return
    
    if (not bool(sdi.getFlags("dataReady"))):
        amplitudeSN = indata[:sampleSize,0]
        sdi.updateBufferArray("amplitudeSN", amplitudeSN)
        sdi.setFlags("dataReady", True)        
        sdi.setFlags("bufferOverrun", False)
    else:
        sdi.setFlags("bufferOverrun", True)

# def bufferProcessor(sdi):
#     sdi.setFlags("processorRunning", True)
#     sdi.setFlags("dataReady", False)

#     while (sdi.getFlags("continueLooping")):
#         if (sdi.getFlags("bufferReady")):
#             if (not sdi.getFlags("dataReady")):  ## did Thread process FFT & lockin data?
#                 filterData (sdi)
#                 getNoiseFloor (sdi)
#                 sdi.setFlags("bufferReady", False)
#                 lockInAmplifier (sdi)
#                 performFFTonData (sdi)
#                 sdi.setFlags("dataReady", True)
#             else:
#                 print ("bufferProcessor: bufferOverrun")
#                 sdi.setFlags("bufferOverrun", True)
#                 sdi.setFlags("bufferReady", False)

#     sdi.setFlags("processorRunning", False)

def soundReceiver(sdi):
    sdi.setFlags("collectorRunning", True)
    sdi.setFlags("bufferReady", False)

    sd.default.samplerate = sdi.getFrequency ("sampleFrequency")
    sd.default.device = sdi.getDeviceInfo("currentDeviceIndex")
    sd.default.channels = 1
    sd.default.dtype = 'int32'   

    counter =  0
    sampleSize = sdi.getBufferInfo("sampleSize")
    samplesToProcess = sdi.getBufferInfo("samplesToProcess")
    sampleFrequency = sdi.getFrequency("sampleFrequency")
    duration = 1.1 * sampleSize / sampleFrequency

    while (sdi.getFlags("continueLooping")):
        indata = sd.rec(int(duration * sampleFrequency), dtype = 'int32', blocking=True)
        amplitudeSN = indata[:sampleSize,0]
        
        if (not sdi.getFlags("bufferReady")):
            sdi.updateBufferArray("amplitudeSN", amplitudeSN)
            sdi.setFlags("bufferReady", True)
            counter += 1
        else:
            print ("soundReceiver: bufferOverrun")
            sdi.setFlags("bufferOverrun", True)

        if (counter == sampleSize or counter == samplesToProcess):
            sdi.setFlags("continueLooping", False)
            print ("Counter exceeds sampleSize or samplesToProcess: ", counter)

    sdi.setFlags("collectorRunning", False)

def generateData(sdi):
    sampleSize = sdi.getBufferInfo("sampleSize")
    freq = sdi.getFrequency("frequency")
    rate = sdi.getFrequency("sampleFrequency")
    endTime = sampleSize / rate 
    theta = 0
    timeBuffer = np.arange(0, endTime, 1/rate)

    # Amplitude of the sine wave is sine of a variable like time
    sinBuff = np.sin(2 * np.pi * freq * timeBuffer + theta)
    cosBuff = np.cos(2 * np.pi * freq * timeBuffer + theta)

    sdi.updateBufferArray("amplitudeSine", sinBuff)
    sdi.updateBufferArray("amplitudeCosine", cosBuff)

    # plot.clf()
    # plot.plot(range(500), sinBuff[500:1000], 'o-r')
    # plot.plot(range(500), cosBuff[500:1000], "k8-")
    # plot.title('Orig Signal')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.legend(['Sine', 'Cosine'])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()  

def getNoiseFloor (sdi):

    if (sdi.getFlags("rawData")):
        sdi.setFFTInfo ("fftThreshold", 0)
        sdi.setLockinInfo ("lockInThreshold", 0)
        return

    filteredNotchedSN = signal.filtfilt(sdi.getFilterCoefficients ("notchB"), 
                                        sdi.getFilterCoefficients ("notchA"), 
                                        sdi.getBufferArray("amplitudeSN"))
    sampleFrequency = sdi.getFrequency("sampleFrequency")
    
    # amplitudeSN = sdi.getBufferArray("amplitudeSN")
    # plot.clf()
    # plot.plot(range(500), amplitudeSN[500:1000], 'o-r')
    # plot.plot(range(500), filteredNotchedSN[500:1000], "k8-")
    # plot.title('GetNoiseFloor Orig Signal')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.legend(['Orig', 'Notch Filter'])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()  

    #########################
    # DDR need to set parameter for standard deviation for FFT threshold
    #########################
    # fftSN = np.fft.fft(amplitudeSN, len(amplitudeSN))
    # SNPower = np.absolute(np.sqrt(fftSN * np.conj(fftSN))/len(fftSN))
    # fftThreshold = np.mean(SNPower) + 4 * np.std(SNPower)
    # freq = np.fft.fftfreq(amplitudeSN.shape[-1],d=1/sampleFrequency)
    # length = np.arange(1,np.floor(len(amplitudeSN)/2), dtype="int")

    # plot.clf()
    # plot.scatter(freq[length], SNPower[length], c ="r", linewidths = 1, marker ="o", edgecolor ="green", s = 10, alpha=0.5)
    # plot.title('GetNoiseFloor Originl Signal FFT Spectrum')
    # plot.xlabel('Freq')
    # plot.ylabel('Power')
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()

##########  FFT Threshold Value
    fftSN = np.fft.fft(filteredNotchedSN, len(filteredNotchedSN))
    SNPower = np.absolute(np.sqrt(fftSN * np.conj(fftSN))/len(fftSN))
    freq = np.fft.fftfreq(filteredNotchedSN.shape[-1],d=1/sampleFrequency)
    length = np.arange(1,np.floor(len(filteredNotchedSN)/2), dtype="int")

    fftThresholdSDMultiplier = sdi.getFFTInfo ("fftThresholdSDMultiplier")
    meanPower = np.mean(SNPower)
    stdPower = np.std(SNPower)
    if (stdPower <= meanPower):
        fftThreshold = meanPower + fftThresholdSDMultiplier * stdPower
    else:
        fftThreshold = meanPower + fftThresholdSDMultiplier * meanPower

    sdi.setFFTInfo ("fftThreshold", fftThreshold)

    # sdi.setFFTInfo ("fftThreshold", fftThreshold)
    # plot.clf()
    # plot.plot(freq[length], SNPower[length], "r")
    # thres = SNPower
    # thres.fill(fftThreshold)
    # plot.plot(freq[length], thres[length], "k") 
    # plot.title('GetNoiseFloor Notch Filtered FFT Spectrum')
    # plot.xlabel('Freq')
    # plot.ylabel('Power')
    # plot.legend (["FFT", "Thresh"])
    # #plot.legend (["FFTorig", "FFT", "Threshold"])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()


##########  Lockin Threshold Value
    #########################
    # DDR need to set parameter for standard deviation for lockin threshold
    #########################    
    averageLockInAmlitude, lockInStandardDevition = getLockInAmplifierMagnitude (filteredNotchedSN, sdi)
    lockinThresholdSDMultiplier = sdi.getLockinInfo ("lockinThresholdSDMultiplier")

    if (lockInStandardDevition <= averageLockInAmlitude):
        lockInThreshold = averageLockInAmlitude + lockinThresholdSDMultiplier*lockInStandardDevition 
    else:
        lockInThreshold = averageLockInAmlitude + lockinThresholdSDMultiplier*averageLockInAmlitude 

    sdi.setLockinInfo ("lockInThreshold", lockInThreshold)

    # print ("Lockin Thresh: ", lockInThreshold)
    # plot.clf()
    # plot.plot(range(len(lockInAmlitude)), lockInAmlitude, 'o-r')
    # thres = lockInAmlitude
    # thres.fill(lockInThreshold)
    # plot.plot(range(len(lockInAmlitude)), thres, "k") 
    # plot.title('GetNoiseFloor Lockin Amp')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.legend(['Lockin', 'Thresh'])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show() 

def notchFilterSetup(sdi):
    Q = 4
    b, a = signal.iirnotch(sdi.getFrequency("frequency"), Q, sdi.getFrequency("sampleFrequency"))
    sdi.setFilterCoefficients ("notchB", b)
    sdi.setFilterCoefficients ("notchA", a)

def chebyFilterSetup(sdi):
    frequency = sdi.getFrequency("frequency")
    sampleFrequency = sdi.getFrequency("sampleFrequency")
    bandwidth = 400 # was 400
    lowPassBandFrequency = sdi.getFrequency("frequency") - bandwidth/2
    highPassBandFrequency = frequency + bandwidth/2
    lowCutoffBandFrequency = lowPassBandFrequency - 200
    highCutoffBandFrequency = highPassBandFrequency + 200

    # Pass band frequency in Hz
    passband = np.array([lowPassBandFrequency, highPassBandFrequency])  
    # Stop band frequency in Hz
    stopband = np.array([lowCutoffBandFrequency, highCutoffBandFrequency])  
    # Pass band ripple in dB
    passbandRipple = 0.4
    # Stop band attenuation in dB
    stopbandAttenution = 50 
    # Normalized passband edge 
    # frequencies w.r.t. Nyquist rate
    normalizedPassband = passband/(sampleFrequency/2)  
    # Normalized stopband 
    # edge frequencies
    normalizedStopband = stopband/(sampleFrequency/2)
    # Compute order of the Chebyshev type-2
    # digital filter using signal.cheb2ord
    order, cutoffFrequencies = signal.cheb2ord(normalizedPassband, normalizedStopband, passbandRipple, stopbandAttenution)
    sos = signal.cheby2(order, stopbandAttenution, cutoffFrequencies, 'bandpass', output='sos')
    sdi.setFilterCoefficients ("sosChebyBP", sos)

    passbandFrequency = 40  # ws 50
    stopbandFrequency = 100 # was 150
    passbandLoss = 0.4
    stopbandLoss = 50 
    #Analog filters need radians per second w=2*pie*F
    #digital filters need fraction with respect to sampling frequency f = f/(Fc/2)
    normalizedPassbandFrequency = passbandFrequency / (sampleFrequency/2)
    normalizedStopbandFrequency = stopbandFrequency / (sampleFrequency/2) 
    order, cutoffFrequencies = signal.cheb2ord(normalizedPassbandFrequency, normalizedStopbandFrequency, gpass=passbandLoss, 
                                               gstop=stopbandLoss, analog=False, fs=sampleFrequency)
    sos = signal.cheby2(order, stopbandLoss, normalizedStopbandFrequency, btype='low', analog=False, output='sos')
    sdi.setFilterCoefficients ("sosChebyDC", sos)

def filterData (sdi):
    bpFilteredSN = signal.sosfiltfilt(sdi.getFilterCoefficients ("sosChebyBP"), sdi.getBufferArray ("amplitudeSN"))
    #lpFilteredSN = signal.sosfiltfilt(sosLow, amplitudeSN)
    sdi.updateBufferArray ("bpFilteredSN", bpFilteredSN)

    # plot.clf()
    # plot.plot(range(500), amplitudeSN[500:1000], 'o-r')
    # plot.plot(range(500), bpFilteredSN[500:1000], "k8-")
    # plot.title('FilterData')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.legend(['Orig', 'BPF'])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()  

def lockInAmplifier (sdi):

    frequencyDelta = sdi.getFrequency("frequencyDelta")
    frequency = sdi.getFrequency("frequency")
    samplerate = sdi.getFrequency("sampleFrequency")

    maxLockIn = 0
    maxFreq = 0

    sdi.setLockinInfo("lockInThresholdFound",  0)
    sdi.setLockinInfo("lockInStandardDevition",  0)

    loopIncrement = 20
    loopCounter = frequencyDelta/loopIncrement 
    testFrequency = frequency + frequencyDelta/2
    
    while (loopCounter > 0):
        if (frequencyDelta > 0):
            sdi.setFrequency("frequency", testFrequency)
            generateData(sdi)
        else:
            loopCounter = -1

        averageLockInAmlitude, lockInStandardDevition = getLockInAmplifierMagnitude (sdi.getBufferArray("bpFilteredSN"), sdi)

        lockInThresholdFound = 0
        lockInThreshold = sdi.getLockinInfo("lockInThreshold")
        sdi.setLockinInfo("lockInFrequency",  0)

        if (sdi.getFlags("rawData")):
            if averageLockInAmlitude > maxLockIn:
                maxLockIn = averageLockInAmlitude
                maxFreq = testFrequency
                sdi.setLockinInfo("lockInThresholdFound",  averageLockInAmlitude)
                sdi.setLockinInfo("lockInStandardDevition",  lockInStandardDevition)
                sdi.setLockinInfo("lockInFrequency",  testFrequency)
        else:
            if averageLockInAmlitude >= lockInThreshold:
                lockInThresholdFound = averageLockInAmlitude
                sdi.setLockinInfo("lockInThresholdFound",  lockInThresholdFound)
                sdi.setLockinInfo("lockInStandardDevition",  lockInStandardDevition)
                sdi.setLockinInfo("lockInFrequency",  testFrequency)
                loopCounter = -1

        loopCounter -= 1
        testFrequency -= loopIncrement

    sdi.setFrequency("frequency", frequency)
    
def getLockInAmplifierMagnitude (signalOfInterest, sdi):

    sineLockInAmlitude = signalOfInterest * sdi.getBufferArray("amplitudeSine")
    cosineLockInAmlitude = signalOfInterest * sdi.getBufferArray("amplitudeCosine")
    sosChebyDC = sdi.getFilterCoefficients("sosChebyDC")
    filteredSineLockInAmlitude = signal.sosfiltfilt(sosChebyDC, sineLockInAmlitude)
    filteredCosineLockInAmlitude = signal.sosfiltfilt(sosChebyDC, cosineLockInAmlitude)

    lockInAmlitude = np.sqrt( (filteredSineLockInAmlitude*filteredSineLockInAmlitude + filteredCosineLockInAmlitude*filteredCosineLockInAmlitude) )

    startingPoint = int(len(lockInAmlitude)/2)
    endingPoint = int(3*len(lockInAmlitude)/4)

    averageLockInAmlitude = np.mean(lockInAmlitude[startingPoint:endingPoint])
    lockInStandardDevition = np.std(lockInAmlitude[startingPoint:endingPoint])

    # plot.clf()
    # plot.plot(range(len(lockInAmlitude)), lockInAmlitude, 'o-r')
    # thres = lockInAmlitude
    # thres.fill(averageLockInAmlitude)
    # plot.plot(range(len(lockInAmlitude)), thres, "k") 
    # plot.title('getlockin')
    # plot.xlabel('Time')
    # plot.ylabel('Amplitude')
    # plot.legend(['Lockin', 'Thres'])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show() 

    return averageLockInAmlitude, lockInStandardDevition

def performFFTonData (sdi):
    sampleFrequency = sdi.getFrequency("sampleFrequency")
    frequency = sdi.getFrequency("frequency")
    bpFilteredSN = sdi.getBufferArray("bpFilteredSN")
    fftThreshold = sdi.getFFTInfo("fftThreshold")
    frequencyDelta = sdi.getFrequency("frequencyDelta")

    fftSN = np.fft.fft(bpFilteredSN, len(bpFilteredSN))
    SNPower = np.absolute(np.sqrt(fftSN * np.conj(fftSN))/len(fftSN))

    freq = np.fft.fftfreq(bpFilteredSN.shape[-1],d=1/sampleFrequency)
    # length = np.arange(1,np.floor(len(bpFilteredSN)/2), dtype="int")
    
    bandwidth = 200
    idxLow3dB = (np.abs(freq - (frequency-bandwidth))).argmin()
    idxHigh3dB = (np.abs(freq - (frequency+bandwidth))).argmin()

    # averageSignal = np.mean(SNPower[idxLow3dB:idxHigh3dB])
    maxValue = SNPower[idxLow3dB:idxHigh3dB].max()
    maxIdx = idxLow3dB - 1 + np.where(SNPower[idxLow3dB:idxHigh3dB] == maxValue)[0]
    minValue = SNPower[idxLow3dB:idxHigh3dB].min()
    # minIdx = idxLow3dB - 1 + np.where(SNPower[idxLow3dB:idxHigh3dB] == minValue)[0]

    fftFoundFrequency = 0.0
    fftFoundPower = 0.0 

    freqDelta = abs(float(freq[maxIdx]) - frequency)

    if (sdi.getFlags("rawData")):
        fftFoundFrequency = float(freq[maxIdx])
        fftFoundPower = np.abs(maxValue)
    else: 
        if (maxValue > fftThreshold and freqDelta <= frequencyDelta):  # within x Hz of main frequency (x Hz above or x Hz below)
            fftFoundFrequency = float(freq[maxIdx])
            fftFoundPower = np.abs(maxValue)

    sdi.setFFTInfo ("fftFoundFrequency", fftFoundFrequency )
    sdi.setFFTInfo ("fftFoundPower", fftFoundPower )

    # plot.clf()
    # plot.scatter(freq[length], SNPower[length], c ="k", linewidths = 2, marker ="p", edgecolor ="orange", s = 10, alpha=0.5)
    # thres = SNPower
    # thres.fill(fftThreshold)
    # plot.plot(freq[length], thres[length], "k8-") 
    # plot.title('PerformFFTonData FFT Spectrum')
    # plot.xlabel('Freq')
    # plot.ylabel('Power')
    # plot.legend (["FFT", "Threshold"])
    # plot.grid(True, which='both')
    # plot.axhline(y=0, color='k')
    # plot.show()

class _Const(object):
    CONNECTED = int("0x00000001", 0)
    SEND_COMMAND = int("0x00000002", 0)
    PLOT = int("0x00000004", 0)
    SAVE = int("0x00000008", 0)
    ENCODING = 'utf-8'

    SETUP_PLOT = 1
    SETUP_TICKS = 2
    NOTHING = 0
    MINIMUM_ENTRIES = 2

    FREQ_700 = "700"    
    FREQ_900 = "900"
    FREQ_1100 = "1100"
    FREQ_1300 = "1300"
    FREQ_1500 = "1500"
    FREQ_1700 = "1700"
    FREQ_1900 = "1900"
    FREQ_2100 = "2100"
    FREQ_2300 = "2300"
    FREQ_2500 = "2500"
    FREQ_2700 = "2700"
    FREQ_2900 = "2900"

if __name__ == '__main__':
    screenHeight = 300
    screenWidth = 200
    nrows = 25
    ncolumns = 50
    flags = 0
    banner = "VE3OOI Signal Detector v0.1"

##################################################
# Window Creation
##################################################
    win = tb.Window(themename='superhero')

#    win.geometry(f"{screenWidth}x{screenHeight}")
    win.title(banner)
    win.protocol("WM_DELETE_WINDOW", closeWindow)

    mainFrame = tb.Frame(win)
    mainFrame.pack()

    statusFrame = tb.LabelFrame(mainFrame, text="Status")
    statusFrame.grid(row=0, column=0, sticky="NSEW")
    statusFrame.grid_columnconfigure(0, weight=1)

    chartFrame = tb.LabelFrame(mainFrame, text="Charts")
    chartFrame.grid(row=1, column=0, sticky="NSEW")
    chartFrame.grid_columnconfigure(0, weight=1)

    # controlFrame = tb.LabelFrame(mainFrame, text="Controls")
    # controlFrame.grid(row=3, column=0, sticky="NSEW")
    # controlFrame.grid_columnconfigure(0, weight=1)

    controlRow1Frame = tb.LabelFrame(mainFrame, text="Controls")
    controlRow1Frame.grid(row=2, column=0, sticky="NSEW")
    controlRow1Frame.grid_columnconfigure(6, weight=1)

    controlRow2Frame = tb.LabelFrame(mainFrame, text="Sound Device")
    controlRow2Frame.grid(row=3, column=0, sticky="NSEW")
    controlRow2Frame.grid_columnconfigure(6, weight=1)

    controlRow3Frame = tb.LabelFrame(mainFrame, text="Configuration")
    controlRow3Frame.grid(row=4, column=0, sticky="NSEW")
    controlRow3Frame.grid_columnconfigure(6, weight=1)

    fillterFrame = tb.LabelFrame(mainFrame, text="Controls4")
    fillterFrame.grid(row=0, column=6, rowspan=5, sticky="NSEW")
    fillterFrame.grid_columnconfigure(6, weight=1)




    def resource_path(relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    photo = ImageTk.PhotoImage(
    Image.open(resource_path("Shack_Icon.png")))
    win.iconphoto(False, photo)
    #win.geometry('600x520')
    #win.resizable(False, False)

    CONST = _Const()

    mainWindow()

    win.mainloop()