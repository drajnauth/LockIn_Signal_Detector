import numpy as np
import matplotlib.pyplot as plot
from scipy import signal 
from scipy import stats
import time as timer

def main():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose, singleFrequency
    
    singleFrequency = 0
    plotting = 0
    fftThreshold = 0
    lockInThreshold = 0
    lockInStandardDevition = 0

# Set parameters here
    verbose = 0 # No output
#    verbose = 1  # Generate diagnostic dataFiltered LockIn Max:
    plotting = 1
    singleFrequency = 0
#    plotting = 1 # Plot signals
#    plotting = 2 # Plot BP Filtered Data
#    plotting = 4 # Plot DC LP Filtered Data
#    plotting = 8 # Plot FFT Spectrum
#    plotting = 16 # Plot Lockin Data

    sampleFrequency = 24000  
    maxSignalAmplitude = 16
    signalDCOffset = 0
    maxNoiseAmplitude = maxSignalAmplitude*2
    noiseDCOffset = 0

    generateData()
    notchFilterSetup()
    chebyFilterSetup()
    butterFilterSetup()     
    filterData () 
    #getNoiseFloor()
    performFFTonData ()
    lockInAmplifier ()

    if fftFoundFrequency > 0 or lockInThresholdFound > 0:
        print("===========")
        print ("Noise Amp: ", maxNoiseAmplitude, " Noise DC: ", noiseDCOffset, " Sig Amp: ", maxSignalAmplitude, " Sig DC: ", signalDCOffset)
        print ("FFT Thresh: ", round(fftThreshold,3) , " Lockin Thresh: ", round(lockInThreshold, 3))

    if fftFoundFrequency > 0:
        if fftFoundFrequency > 1300 and fftFoundFrequency < 1400:
            print ("FFT found Freq: ", round(fftFoundFrequency,0), " with power: ", round(fftFoundPower,3))
        else:
            print ("FFT found suspect Freq: ", round(fftFoundFrequency,0), " with power: ", round(fftFoundPower,3))

    if lockInThresholdFound > 0:
        print ("Lockin Found Signal with amplitude: ", round(lockInThresholdFound,3), " Signal original amplitude: ", round(maxSignalAmplitude, 3), " with SRD: ", round(lockInStandardDevition,3) )

    if fftFoundFrequency > 0 or lockInThresholdFound > 0:
        print ()

    '''
    print ("maxNoiseAmplitude, maxSignalAmplitude, fftThreshold, lockInThreshold", end=', ')
    print ("fftFoundFrequency, fftFoundPower, lockInThresholdFound, lockInStandardDevition" )

    for x in range(30):
        maxSignalAmplitude = 8
        signalDCOffset = 0
        maxNoiseAmplitude = maxSignalAmplitude*x
        noiseDCOffset = 0

        generateData()
        notchFilterSetup()
        chebyFilterSetup()
        butterFilterSetup()    
        getNoiseFloor()
        filterData ()
        performFFTonData ()
        lockInAmplifier ()
        if fftFoundFrequency > 0 or lockInThresholdFound > 0:
            print (maxNoiseAmplitude, ", ", maxSignalAmplitude, ", ", round(fftThreshold,3) , "," , round(lockInThreshold, 3), end=', ')
            print (round(fftFoundFrequency,0), ", ", round(fftFoundPower,3), ", ", round(lockInThresholdFound,3), " , ", round(lockInStandardDevition,3) )
    '''
    #    filterResponse ()
    #    filterPerformance()
    #    fftPerformance()

def filterResponse ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

    butterFilterSetup()
    chebyFilterSetup()

    # Cheby LPF
    wz, hz = signal.sosfreqz(sosChebyLow)
    Mag = 20*np.log10(abs(hz))
    Freq = wz*sampleFrequency/(2*np.pi)
    plot.plot(Freq, Mag, 'r', linewidth=2)

    # Butter LPF
    wz, hz = signal.sosfreqz(sosLow)
    Mag = 20*np.log10(abs(hz))
    Freq = wz*sampleFrequency/(2*np.pi)
    plot.plot(Freq, Mag, 'b', linewidth=2)

    plot.title('LPF Magnitude Response', fontsize=20)
    plot.xlabel('Frequency [Hz]', fontsize=20)
    plot.ylabel('Magnitude [dB]', fontsize=20)
    plot.grid()
    plot.show()


    # Cheby BPF
    wz, hz = signal.sosfreqz(sosChebyBP)
    Mag = 20*np.log10(abs(hz))
    Freq = wz*sampleFrequency/(2*np.pi)
    plot.plot(Freq, Mag, 'r', linewidth=2)

    # Butter LPF
    wz, hz = signal.sosfreqz(sosBP)
    Mag = 20*np.log10(abs(hz))
    Freq = wz*sampleFrequency/(2*np.pi)
    plot.plot(Freq, Mag, 'b', linewidth=2)

    plot.title('BPF Magnitude Response', fontsize=20)
    plot.xlabel('Frequency [Hz]', fontsize=20)
    plot.ylabel('Magnitude [dB]', fontsize=20)
    plot.grid()
    plot.show()

     # Cheby BPF
    wz, hz = signal.sosfreqz(sosChebyDC)
    Mag = 20*np.log10(abs(hz))
    Freq = wz*sampleFrequency/(2*np.pi)
    plot.plot(Freq, Mag, 'r', linewidth=2)

    # Butter LPF
    wz, hz = signal.sosfreqz(sosDC)
    Mag = 20*np.log10(abs(hz))
    Freq = wz*sampleFrequency/(2*np.pi)
    plot.plot(Freq, Mag, 'b', linewidth=2)

    plot.title('DC Magnitude Response', fontsize=20)
    plot.xlabel('Frequency [Hz]', fontsize=20)
    plot.ylabel('Magnitude [dB]', fontsize=20)
    plot.grid()
    plot.show()

def filterPerformance ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

    butterFilterSetup()
    chebyFilterSetup()

    ### LPF 
    startTime = timer.time()
    filteredAmplitudeSignal = signal.sosfiltfilt(sosLow, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Butter LP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    startTime = timer.time()
    chebyFilteredAmplitudeSignal = signal.sosfiltfilt(sosChebyLow, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Cheby LP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    plot.scatter(time[:len(filteredAmplitudeSignal)], filteredAmplitudeSignal, c ="b", linewidths = 2, marker ="p", edgecolor ="red", s = 10, alpha=0.5)
    plot.scatter(time[:len(chebyFilteredAmplitudeSignal)], chebyFilteredAmplitudeSignal, c ="r", linewidths = 2, marker ="D", edgecolor ="blue", s = 10, alpha=0.5)

    plot.title('LPF Filter comparison')
    plot.xlabel('Freq')
    plot.ylabel('Amplitude')
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.show()

    ## BPF
    startTime = timer.time()
    filteredAmplitudeSignal = signal.sosfiltfilt(sosBP, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Butter BP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    startTime = timer.time()
    chebyFilteredAmplitudeSignal = signal.sosfiltfilt(sosChebyBP, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Cheby BP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    plot.scatter(time[:len(filteredAmplitudeSignal)], filteredAmplitudeSignal, c ="b", linewidths = 2, marker ="p", edgecolor ="red", s = 10, alpha=0.5)
    plot.scatter(time[:len(chebyFilteredAmplitudeSignal)], chebyFilteredAmplitudeSignal, c ="r", linewidths = 2, marker ="D", edgecolor ="blue", s = 10, alpha=0.5)

    plot.title('BP Filter comparison')
    plot.xlabel('Freq')
    plot.ylabel('Amplitude')
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.show()

    ## DC
    startTime = timer.time()
    filteredAmplitudeSignal = signal.sosfiltfilt(sosDC, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Butter DC Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    startTime = timer.time()
    chebyFilteredAmplitudeSignal = signal.sosfiltfilt(sosChebyDC, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Cheby DC Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    plot.scatter(time[:len(filteredAmplitudeSignal)], filteredAmplitudeSignal, c ="b", linewidths = 2, marker ="p", edgecolor ="red", s = 10, alpha=0.5)
    plot.scatter(time[:len(chebyFilteredAmplitudeSignal)], chebyFilteredAmplitudeSignal, c ="r", linewidths = 2, marker ="D", edgecolor ="blue", s = 10, alpha=0.5)

    plot.title('DC Filter comparison')
    plot.xlabel('Freq')
    plot.ylabel('Amplitude')
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.show()

def fftPerformance ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

    butterFilterSetup()
    chebyFilterSetup()

    filterData()

    startTime = timer.time()
    bpFilteredSN = signal.sosfiltfilt(sosChebyBP, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Cheby BP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    startTime = timer.time()
    fftSN = np.fft.fft(bpFilteredSN, len(bpFilteredSN))
    SNPower = np.sqrt(fftSN * np.conj(fftSN))/len(fftSN)
    SNPower = np.abs(SNPower)
    endTime = timer.time()
    if verbose:
        print("Cheby FFT Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")
    freq = np.fft.fftfreq(bpFilteredSN.shape[-1],d=1/sampleFrequency)
    length = np.arange(1,np.floor(len(bpFilteredSN)/2), dtype="int")


    bandwidth = 200
    idxLow3dB = (np.abs(freq - (frequency-bandwidth))).argmin()
    idxHigh3dB = (np.abs(freq - (frequency+bandwidth))).argmin()

    averageSignal = np.mean(SNPower[idxLow3dB:idxHigh3dB])
    maxValue = SNPower[idxLow3dB:idxHigh3dB].max()
    maxIdx = idxLow3dB - 1 + np.where(SNPower[idxLow3dB:idxHigh3dB] == maxValue)[0]
    minValue = SNPower[idxLow3dB:idxHigh3dB].min()
    minIdx = idxLow3dB - 1 + np.where(SNPower[idxLow3dB:idxHigh3dB] == minValue)[0]
    if verbose:
        print ("Signal Max: ", maxValue, " at index: ", maxIdx, " Freq: ", float(freq[maxIdx]) )
        print ("Signal Min: ", minValue, " at index: ", minIdx, " Freq: ", float(freq[minIdx]) )
        print ("Signal Avg (DC): ", averageSignal)    

    if maxValue > fftThreshold:
        fftFoundFrequency = float(freq[maxIdx])
        fftFoundPower = maxValue
        if verbose:
            print ("Signal present at Freq: ", fftFoundFrequency, " with power: ", fftFoundPower) 
    
    if plotting == 8:
        plot.scatter(freq[length], SNPower[length], c ="b", marker ="^")    
        thres = SNPower
        thres.fill(fftThreshold)
        plot.plot(freq[length], thres[length], "k8-") 
        plot.title('FFT Spectrum')
        plot.xlabel('Freq')
        plot.ylabel('Amplitude')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()  
    
def getNoiseFloor ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

######################### Noise
    
    startTime = timer.time()
    filteredNoise_rnd = signal.filtfilt(notchB, notchA, noise_rnd)
    endTime = timer.time()
    if verbose:
        print("Notch Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")


    startTime = timer.time()
    fftSN = np.fft.fft(filteredNoise_rnd, len(filteredNoise_rnd))
    SNPower = np.sqrt(fftSN * np.conj(fftSN))/len(fftSN)
    endTime = timer.time()
    if verbose:
        print("RND Noise FFT Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    averageNoise = np.mean(SNPower)
    if verbose:
        print ("Noise RND FFT Max: ", abs(SNPower.max()))
        print ("Noise RND FFT Min: ", abs(SNPower.min()))
        print ("Noise RND FFT Avg: ", averageNoise)
        print ("Noise RND FFT STD: ", np.std(SNPower))
    fftThreshold = np.real(averageNoise) + 4 * np.real(np.std(SNPower))

    filteredNoise_sd = signal.filtfilt(notchB, notchA, noise_sd)
    startTime = timer.time()
    fftSN = np.fft.fft(filteredNoise_sd, len(filteredNoise_sd))
    SNPower = np.sqrt(fftSN * np.conj(fftSN))/len(fftSN)
    endTime = timer.time()
    if verbose:
        print("SD Noise FFT Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    if verbose:
        averageNoise = np.mean(SNPower)
        print ("Noise SD FFT Max: ", SNPower.max())
        print ("Noise SD FFT Min: ", SNPower.min())
        print ("Noise SD FFT Avg: ", averageNoise)
        print ("Noise SD FFT STD: ", np.std(SNPower))

        print ("FFT Threshold: ", fftThreshold)

##########  lockin Value

    noiseLockinAmplitude = getLockInAmplifierMagnitude (filteredNoise_rnd)

    startTime = timer.time()
    filteredNotchedSN = signal.filtfilt(notchB, notchA, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("Notch Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    lockInThreshold = getLockInAmplifierMagnitude (filteredNotchedSN)

    if verbose:
        print("LockIn Noise Amplitude:  ", noiseLockinAmplitude)
        print("LockIn Signal with Notch Threshold:  ", lockInThreshold)
        print("LockIn Signal with Notch STD:  ", lockInStandardDevition)

def generateData ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose


# Get x values of the sine wave
    frequency = 1300
    sampleFrequency = 24000
    dataPointPerPeriod = int(sampleFrequency/frequency)
    totalSamplesPerBuffer = 100
#    sampleDataPoints = totalSamplesPerBuffer * dataPointPerPeriod
    sampleDataPoints = 10000
    sampleStepSize = 2 * np.pi / dataPointPerPeriod
    maximumLength = 8192

    phaseShift = int(dataPointPerPeriod/4)

    if verbose:
        print ("Sample Frequency: "+str(sampleFrequency))
        print ("Data Points/Period: "+str(dataPointPerPeriod))
        print ("Data Points: "+str(sampleDataPoints))
        print ("Step Size: "+str(sampleStepSize))

    time = np.arange(0, sampleDataPoints, sampleStepSize)
    timePhaseShift = np.arange(0, sampleDataPoints+phaseShift, sampleStepSize)

    # Amplitude of the sine wave is sine of a variable like time
    amplitudeSine   = np.sin(time[:maximumLength])
    amplitudeCosine   = np.cos(time[:maximumLength])
 
    if verbose:
        print ("Signal 1 Frequency: ", frequency, " amplitude: ", maxSignalAmplitude, " DC offset: ", signalDCOffset)
    amplitudeSignalPhaseShifted   = maxSignalAmplitude*np.sin(timePhaseShift) + signalDCOffset
    extraLength = len(amplitudeSignalPhaseShifted) - len(amplitudeSine)
    amplitudeSignal1 = amplitudeSignalPhaseShifted[extraLength:]

    if len(amplitudeSignal1) != len(amplitudeSine):
        print ("Arrays not equal in length")
        exit

    if (not singleFrequency):
        frequency = 2300
        sampleFrequency = 24000
        dataPointPerPeriod = int(sampleFrequency/frequency)
        totalSamplesPerBuffer = 100
    #    sampleDataPoints = totalSamplesPerBuffer * dataPointPerPeriod
        sampleDataPoints = 10000
        sampleStepSize = 2 * np.pi / dataPointPerPeriod
        time = np.arange(0, sampleDataPoints, sampleStepSize)
        amplitudeSignal2   = maxSignalAmplitude*np.sin(time) + signalDCOffset
        if verbose:
            print ("Signal 2 Frequency: ", frequency, " amplitude: ", maxSignalAmplitude, " DC offset: ", signalDCOffset)

        frequency = 1000
        sampleFrequency = 24000
        dataPointPerPeriod = int(sampleFrequency/frequency)
        totalSamplesPerBuffer = 100
    #    sampleDataPoints = totalSamplesPerBuffer * dataPointPerPeriod
        sampleDataPoints = 10000
        sampleStepSize = 2 * np.pi / dataPointPerPeriod
        time = np.arange(0, sampleDataPoints, sampleStepSize)
        amplitudeSignal3   = maxSignalAmplitude*np.sin(time) + signalDCOffset
        if verbose:
            print ("Signal 3 Frequency: ", frequency, " amplitude: ", maxSignalAmplitude, " DC offset: ", signalDCOffset)


        amplitudeSignal = amplitudeSignal1[:maximumLength] + amplitudeSignal2[:maximumLength] + amplitudeSignal3[:maximumLength]

        if verbose:
            averageSignal = np.mean(amplitudeSignal)
            print ("Signal Max: ", amplitudeSignal.max())
            print ("Signal Min: ", amplitudeSignal.min())
            print ("Signal Avg (DC): ", averageSignal)

        noiseMean = noiseDCOffset
        noiseSD = maxNoiseAmplitude
        noise_sd = np.random.normal(noiseMean, noiseSD, len(amplitudeSignal)) 
        #noise = np.random.randn(len(amplitude)) 
        if verbose:
            averageNoise = np.mean(noise_sd)
            print ("Nose Max: ", noise_sd.max())
            print ("Nose Min: ", noise_sd.min())
            print ("Noise Avg (DC): ", averageNoise)

        noise_rnd = maxNoiseAmplitude*np.random.randn(len(amplitudeSignal)) + noiseDCOffset
        if verbose:
            averageNoise = np.mean(noise_rnd)
            print ("Nose Max: ", noise_rnd.max())
            print ("Nose Min: ", noise_rnd.min())
            print ("Noise Avg (DC): ", averageNoise)

        amplitudeSN = amplitudeSignal + noise_rnd
        if verbose:
            averageSN = np.mean(amplitudeSN)
            print ("SN Max: ", amplitudeSN.max())
            print ("SN Min: ", amplitudeSN.min())
            print ("SN Avg: ", averageSN)
    else:
        amplitudeSN = amplitudeSignal1[:maximumLength] * 2


    frequency = 1300
    sampleFrequency = 24000

    if plotting == 1:
        plot.clf()
        #plot.plot(time[:8192], amplitudeSN[:8192], "r^-")
        plot.plot(time[:8192], amplitudeSignal1[:8192], "b^-")
        plot.title('Reference')
        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        #plot.legend(['All Signals', 'Signal'])
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

def butterFilterSetup():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

    sosHigh = signal.butter(4, 600, 'high', fs=sampleFrequency, output='sos')
    sosLow = signal.butter(4, 1350, 'low', fs=sampleFrequency, output='sos')
    sosDC = signal.butter(4, 100, 'low', fs=sampleFrequency, output='sos')

    centerFrequency = frequency
    bandwidth = 400
    lowPassBandFrequency = centerFrequency - bandwidth/2
    highPassBandFrequency = centerFrequency + bandwidth/2
    lowCutoffBandFrequency = lowPassBandFrequency - 200
    highCutoffBandFrequency = highPassBandFrequency + 200    
    sosBP = signal.butter(4, [lowCutoffBandFrequency, highCutoffBandFrequency], 'bandpass', fs=sampleFrequency, output='sos')

def notchFilterSetup():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

    Q = 2
    notchB, notchA = signal.iirnotch(frequency, Q, sampleFrequency)
    print ("notchB shape", notchB.shape)
    print ("notchA shape", notchA.shape)

def chebyFilterSetup():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose


    centerFrequency = frequency
    bandwidth = 400
    lowPassBandFrequency = centerFrequency - bandwidth/2
    highPassBandFrequency = centerFrequency + bandwidth/2
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
    sosChebyBP = signal.cheby2(order, stopbandAttenution, cutoffFrequencies, 'bandpass', output='sos')
    print ("sosChebyBP shape", sosChebyBP.shape)

    passbandFrequency = 1350
    stopbandFrequency = 1450
    passbandLoss = 0.4
    stopbandLoss = 50 
    #Analog filters need radians per second w=2*pie*F
    #digital filters need fraction with respect to sampling frequency f = f/(Fc/2)
    normalizedPassbandFrequency = passbandFrequency / (sampleFrequency/2)
    normalizedStopbandFrequency = stopbandFrequency / (sampleFrequency/2) 
    order, cutoffFrequencies = signal.cheb2ord(normalizedPassbandFrequency, normalizedStopbandFrequency, gpass=passbandLoss, gstop=stopbandLoss, analog=False, fs=sampleFrequency)
    sosChebyLow = signal.cheby2(order, stopbandLoss, normalizedStopbandFrequency, btype='low', analog=False, output='sos')
    print ("sosChebyLow shape", sosChebyLow.shape)
    passbandFrequency = 50
    stopbandFrequency = 150
    passbandLoss = 0.4
    stopbandLoss = 50 
    #Analog filters need radians per second w=2*pie*F
    #digital filters need fraction with respect to sampling frequency f = f/(Fc/2)
    normalizedPassbandFrequency = passbandFrequency / (sampleFrequency/2)
    normalizedStopbandFrequency = stopbandFrequency / (sampleFrequency/2) 
    order, cutoffFrequencies = signal.cheb2ord(normalizedPassbandFrequency, normalizedStopbandFrequency, gpass=passbandLoss, gstop=stopbandLoss, analog=False, fs=sampleFrequency)
    sosChebyDC = signal.cheby2(order, stopbandLoss, normalizedStopbandFrequency, btype='low', analog=False, output='sos')
    print ("sosChebyDC shape", sosChebyDC.shape)

def filterData ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting

#Remove DC from both Signals
    
    startTime = timer.time()
    bpFilteredSN = signal.sosfiltfilt(sosChebyBP, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("BP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")
   
    startTime = timer.time()
    lpFilteredSN = signal.sosfiltfilt(sosLow, amplitudeSN)
    endTime = timer.time()
    if verbose:
        print("LP Filter Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    if plotting == 2:
        plot.clf()
        plot.plot(time[500:1000], bpFilteredSN[500:1000], 'o-r')
        plot.plot(time[500:1000], amplitudeSN[500:1000], '^-.k')    
        plot.title('BP Filtered Data')
        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        plot.legend(['Filtered', 'All Signals'])
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

def lockInAmplifier ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose

    averageLockInAmlitude = getLockInAmplifierMagnitude (bpFilteredSN)

    lockInThresholdFound = 0
    if averageLockInAmlitude >= maxSignalAmplitude/3 and maxSignalAmplitude > 0:
        lockInThresholdFound = averageLockInAmlitude

def getLockInAmplifierMagnitude (signalOfInterest):
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose


    sineLockInAmlitude = signalOfInterest * amplitudeSine
    cosineLockInAmlitude = signalOfInterest * amplitudeCosine

    startTime = timer.time()
    filteredSineLockInAmlitude = signal.sosfiltfilt(sosChebyDC, sineLockInAmlitude)
    filteredCosineLockInAmlitude = signal.sosfiltfilt(sosChebyDC, cosineLockInAmlitude)

    lockInAmlitude = np.sqrt( (filteredSineLockInAmlitude*filteredSineLockInAmlitude + filteredCosineLockInAmlitude*filteredCosineLockInAmlitude) )
    endTime = timer.time()
    if verbose:
        print("LockIn DC LP Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    startingPoint = int(len(lockInAmlitude)/4)
    endingPoint = int(3*len(lockInAmlitude)/4)
    if verbose:
        print("Checking from element ", startingPoint, " to ", endingPoint)

    averageLockInAmlitude = np.mean(lockInAmlitude[startingPoint:endingPoint])
    lockInStandardDevition = np.std(lockInAmlitude[startingPoint:endingPoint])
    if verbose:
        print ("Filtered LockIn Max: ", lockInAmlitude[startingPoint:endingPoint].max())
        print ("Filtered LockIn Min: ", lockInAmlitude[startingPoint:endingPoint].min())
        print ("Filtered LockIn Avg: ", averageLockInAmlitude)
        print ("Filtered LockIn Mode: ", stats.mode(lockInAmlitude[startingPoint:endingPoint]))
        print ("Filtered LockIn STD: ", np.std(lockInAmlitude[startingPoint:endingPoint]))

    if plotting == 16:
        plot.clf()
        plot.plot(range(endingPoint-startingPoint), lockInAmlitude[startingPoint:endingPoint], 'o-r')
        thres = lockInAmlitude
        thres.fill(averageLockInAmlitude)
        plot.plot(range(endingPoint-startingPoint), thres[startingPoint:endingPoint], "k8-") 
        plot.title('Lockin Results')
        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        plot.legend(['Lockin Data', 'Averge'])
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

    return averageLockInAmlitude

def performFFTonData ():
    global frequency, sampleFrequency, dataPointPerPeriod, totalSamplesPerBuffer, sampleDataPoints
    global maxSignalAmplitude, maxNoiseAmplitude, signalDCOffset, noiseDCOffset
    global time, amplitudeSine, amplitudeCosine
    global amplitudeSignal, noise_sd, noise_rnd, amplitudeSN, bpFilteredSN, lpFilteredSN
    global sosHigh, sosLow, sosDC, sosBP, sosChebyBP, sosChebyLow, sosChebyDC, notchB, notchA
    global fftThreshold, lockInThreshold, lockInStandardDevition, autocorrelateThreshold
    global fftFoundFrequency, fftFoundPower, lockInThresholdFound, autocorrelateThresholdFrequency, autocorrelateThresholdIntensity
    global plotting, verbose


    startTime = timer.time()
    fftSN = np.fft.fft(bpFilteredSN, len(bpFilteredSN))
    SNPower = np.sqrt(fftSN * np.conj(fftSN))/len(fftSN)
    endTime = timer.time()
    if verbose:
        print("FFT Elapsed Execution Time: ", (endTime-startTime)*1000, " mS")

    freq = np.fft.fftfreq(bpFilteredSN.shape[-1],d=1/sampleFrequency)
    length = np.arange(1,np.floor(len(bpFilteredSN)/2), dtype="int")
    
    bandwidth = 200
    idxLow3dB = (np.abs(freq - (frequency-bandwidth))).argmin()
    idxHigh3dB = (np.abs(freq - (frequency+bandwidth))).argmin()

    averageSignal = np.mean(SNPower[idxLow3dB:idxHigh3dB])
    maxValue = SNPower[idxLow3dB:idxHigh3dB].max()
    maxIdx = idxLow3dB - 1 + np.where(SNPower[idxLow3dB:idxHigh3dB] == maxValue)[0]
    minValue = SNPower[idxLow3dB:idxHigh3dB].min()
    minIdx = idxLow3dB - 1 + np.where(SNPower[idxLow3dB:idxHigh3dB] == minValue)[0]
    if verbose:
        print ("Signal Max: ", maxValue, " at index: ", maxIdx, " Freq: ", freq[maxIdx] )
        print ("Signal Min: ", minValue, " at index: ", minIdx, " Freq: ", freq[minIdx] )
        print ("Signal Avg (DC): ", averageSignal)  

    fftFoundFrequency = 0.0
    fftFoundPower = 0.0 

    if maxValue > fftThreshold:
        fftFoundFrequency = float(freq[maxIdx])
        fftFoundPower = np.abs(maxValue)
        if verbose:
            print ("Signal present at Freq: ", fftFoundFrequency, " with power: ", fftFoundPower) 

  
    if plotting == 8:
        plot.clf()
        plot.scatter(freq[length], SNPower[length], c ="k", linewidths = 2, marker ="p", edgecolor ="orange", s = 10, alpha=0.5)
        thres = SNPower
        thres.fill(fftThreshold)
        plot.plot(freq[length], thres[length], "k8-") 
        plot.title('FFT Spectrum')
        plot.xlabel('Freq')
        plot.ylabel('Power')
        plot.legend ("FFT", "Threshold")
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

if __name__ == "__main__":
    main()

'''
    indices = np.arange(0, len(chebyFilteredAmplitudeSignal)/2, 1, dtype=int )
    mx = []
    fq = []
    
    for idx in indices:
        if SNPower[idx] > fftThreshold: 
            fq.append(freq[idx])
            mx.append(SNPower[idx])
    mx = np.array(mx)
    fq = np.array(fq)
    mx = np.absolute(mx)

    averageSignal = np.mean(mx)
    maxValue = mx.max()
    maxIdx = np.where(mx == maxValue)[0]
    minValue = mx.min()
    minIdx = np.where(mx == minValue)[0]
    print ("Sub Signal Max: ", maxValue, " at index: ", maxIdx, " Freq: ", fq[maxIdx] )
    print ("Sub Signal Min: ", minValue, " at index: ", minIdx, " Freq: ", fq[minIdx] )
    print ("Sub Signal Avg (DC): ", averageSignal)

    averageSignal = np.mean(SNPower[length])
    maxValue = SNPower[length].max()
    maxIdx = np.where(SNPower[length] == maxValue)[0]
    minValue = SNPower[length].min()
    minIdx = np.where(SNPower[length] == minValue)[0]
    print ("Signal Max: ", maxValue, " at index: ", maxIdx, " Freq: ", freq[maxIdx] )
    print ("Signal Min: ", minValue, " at index: ", minIdx, " Freq: ", freq[minIdx] )
    print ("Signal Avg (DC): ", averageSignal)

'''



