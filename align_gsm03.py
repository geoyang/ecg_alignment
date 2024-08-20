# Align two corresponding contec and eb file
# Requires contec_ecg file, contec reader contec_gsm91
# G. Stuart Mendenhall, MD
# 8/18/2024
# version 0.3 - Class structiure.  R^2 fixed and only for aligned segment, display routines.
# This utility reads in the Contec ECG file and the everbeat CSV file.  It will search within a padding range
# for alignment of the peaks in the overlapping range.  It does simple peak align based on half max deviation and
# blanks everything below the half max.
# Offset and overall R^2 correlation is returned.
#
# Marker is the press on the contec (first press is 0, second 1)
#
# surfidx is the surface lead to align in the contec recording (0 is lead I surface, 1 is lead II)
# everbeat is read as a single dimension vector

from contec_gsm91 import contec_ecg
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import argparse

if __name__ == "__main__" and "get_ipython" not in dir():
    parser = argparse.ArgumentParser(
        description='Output CSV file from CONTEC TLC5000 ECG file')
    parser.add_argument("reference", help="Reference file.", type=str)
    parser.add_argument("eval", help="Eval file.", type=str)
    args = parser.parse_args()
    reference= args.reference
    eval = args.eval
    print(str(reference) + " eval: "+ str(eval))


class contec_align():


    def __init__(self, ecg_filename, eb_filename):
        
        self.c = contec_ecg(ecg_filename)
        self.eba=self.c.read_eb(eb_filename)
        self.ecga=self.c.ecg_array()
        self.metadata=self.c.read_mrk()


    def align_segments(self, lowpad=3000, highpad=9000, plotrange=9000, marker_index=0):
        # Align the segments read in the initiation
        # Marker_index is which contec marker to use - zero indexed!  First marker press is 0, second is 1...
        # Lowpad and Highpad is the padding range to search for the alignment.  This is INDEPENDENT
        #   of the noise truncation portion.
        # Plotrange is the length of the final aligned segments to return.  NO BOUNDARY CHECK.
        # This is the divisor to convert from the marker index to the contec array index.
        # Should typically be 24 for the contec TCL5000 we are using.
        factor=24

        # Resample the contec ecg to correspond with the 512 Hz everbeat sampling rate
        resamp = self.c.upsample_ecg(self.ecga, 512)
        resamp_mrk = self.c.upsample_mrk(self.metadata, 512)
        print('new length',len(resamp[0]))

        resamp_mrk/factor
        
        # Which contec surface lead to use for alignment - zero indexed!
        surfidx = 0

        # Flip recordings if required
        #eba = -eba
        #resamp_debias = -resamp_debias

        print("Aligning everbeat and contec using Marker ", marker_index, "and surface lead index ",surfidx)

        resamp_mrk_offset = resamp_mrk[marker_index]/factor
        resamp_debias = self.c.debias_leads(resamp)


        # Remove the first part of the noisy everbeat recording
        eb_noise_truncation = 0
        eb_norm = 0.12
        ebn=self.eba*eb_norm
        ebt=ebn[eb_noise_truncation:]
        ebz=np.copy(ebt)
        resamp_debias_zeroed = np.copy(resamp_debias)

        leftrange = int(resamp_mrk_offset+eb_noise_truncation)
        rightrange = int(resamp_mrk_offset+eb_noise_truncation+highpad)

        eb_rangemax = np.max(resamp_debias_zeroed[leftrange:rightrange],surfidx)

        # Zero out all values that are less than half of the max (only weight the high amplitude values)
        ebz[ebz<(np.max(ebt)/2)]=0
        resamp_debias_zeroed[resamp_debias_zeroed<(eb_rangemax/2)]=0

        # Array of the dot products for each offset
        dots=[]
        for offset in range(highpad-lowpad):
        #for offset in range(0):
            # Compare the offset ebz to the surface recording
            leftind = int(resamp_mrk_offset+offset+eb_noise_truncation)
            rightind = int(resamp_mrk_offset+ebz.shape[0]+offset+eb_noise_truncation)
            dotprod = np.dot(resamp_debias_zeroed[leftind:rightind, surfidx], ebz)
            dots.append(dotprod)

        optimal_offset=np.argmax(dots)

        print("Using optimal offest ",optimal_offset)

        #Pad the everbeat with the optimal offset calculated from peak alignment
        #For some reason optimal offset consistently seems just a few values later... unclear reason...
        ebtpadz=np.pad(ebn,pad_width=(optimal_offset-3,0))

        surfl=int(resamp_mrk_offset+optimal_offset+eb_noise_truncation)
        surfr=int(surfl+plotrange)

        ebcleanl=int(optimal_offset+eb_noise_truncation)
        ebcleanr=int(optimal_offset+eb_noise_truncation+plotrange)

        self.cleaneb = np.copy(ebtpadz[ebcleanl:ebcleanr])
        self.cleansurf = np.copy(resamp_debias[surfl:surfr,surfidx])
        self.dots = dots
        return (self.cleaneb, self.cleansurf)
    
    def plot_dots(self):
        plt.figure(figsize=(20,6))
        #Plot the dot product to display the max
        plt.plot(self.dots)
        # plt.show()

    def plot_align(self, left_highlight = 0, right_highlight = 0, offset=-10):
        plt.figure(figsize=(20,6))
        #plt.plot(self.cleaneb)
        #plt.plot(self.cleansurf+offset)
        plt.plot(self.c.debias_leads(self.cleaneb))
        plt.plot(self.c.debias_leads(self.cleansurf)+offset)
        plt.show()
        if left_highlight & right_highlight:
            plt.axvspan(left_highlight,right_highlight, color='cyan', alpha = 0.5)
      

    def calc_r2(self):
        # De-bias again
        debias_aligned_eb = self.c.debias_leads(self.cleaneb)
        debias_aligned_surf = self.c.debias_leads(self.cleansurf)
        #rsq = np.corrcoef(self.cleaneb, self.cleansurf)
        rsq = np.corrcoef(debias_aligned_eb[0:4000], debias_aligned_surf[0:4000])
        return(rsq)

    def assess_quality(self, blanking_width=50):
        # Get the maximum dot product, and find fraction of the "second_best" maximum
        # A lower fraction (<0.9) typically indicates a true "best" alignment
        absolute_max = np.max(self.dots)
        argmax=np.argmax(self.dots)
        leftpad = int(argmax-blanking_width)
        rightpad = int(argmax+blanking_width)
        max_left = np.max(self.dots[0:leftpad])
        max_right = np.max(self.dots[rightpad:])
        competitor = np.max([max_left,max_right])
        q_fract = competitor / absolute_max
        
        print("Quality fraction is ",q_fract)
        if (q_fract<0.9):
            print("\nThe alignment is likely successful")
        else:
            print("\nSince the peak alignment was not very strongly unique, may want to manual check alignment.")
        return(q_fract)

a=contec_align(reference, eval)
a.align_segments(plotrange=15000)
a.plot_align(450,1600)
# a.plot_dots()
a.calc_r2()
a.assess_quality()