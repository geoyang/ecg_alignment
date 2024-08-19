# Parse Contec TCL5000 ECG binary data file
# G. Stuart Mendenhall, MD
# 7/6/2024
# version 0.91 - added resampling, terminal split portion
# This utility reads in the TCL .ecg file and uses .mrk annotation to optionally split along markers
# Output is CSV of values.  Optional upsampling through linear interoploation.

"""contec_tlc_GSM:  Parse the Contec TCL5000 ECG"""


#!/usr/bin/env python
__author__= "G. Stuart Mendenhall, MD"
__copyright__ = "Copyright 2024"


import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import argparse


# Constants for the TCL500
samp_rate = 200
num_leads = 12

# Bias added to the values of all leads - this does not seem to be constant, so would not use.
global_offset=0

# File format
filename_suffix = 'ecg'
dataname_suffix = 'dat'
markname_suffix = 'mrk'
ebname_suffix = 'eb'

# File empiric determinants
endianness = 'little'
intsize_in_bytes = 2
marker_intsize_in_bytes = 4
split_divisor_default = 24

header_size = 0


class contec_ecg():
    def __init__(self, filename, header_size=header_size, num_leads=num_leads):
        self.header_size = header_size
        self.num_leads = num_leads
        self.filename = str(filename)+'.'+filename_suffix
        self.dataname = str(filename)+'.'+dataname_suffix
        self.markname = str(filename)+'.'+markname_suffix
        # For now, eb file is defined in function
        #self.ebname = str(filename)+'.'+ebname_suffix
        self.filesize = os.path.getsize(self.filename)
        
        try:
            file = open(self.filename, 'rb')
        except:
            print("Unable to open file!\n")
            return None
    
        # Dump header for analysis
        for headerint in range(header_size):
            print('Header ', int.from_bytes(file.read(1), endianness))
        file.close()
    
    # READS EB array.  Proprietary function, remove before any public release
    def read_eb(self, ebnum):
        self.ebname = str(ebnum)+'.'+ebname_suffix
        eb_array = np.loadtxt(self.ebname, dtype='int',usecols=0, delimiter=',', converters=float)
        return(eb_array)

    def read_mrk(self, bytesize = marker_intsize_in_bytes):
        file_mrk = open(self.markname,'rb')
        EOF = False
        datavector = []
        data_bytes_read = 0
        while True:
            datapoint = file_mrk.read(bytesize)
            if (len(datapoint) < bytesize):
                print('EOF marker file reached. ')
                EOF = True
                break
            else:
                data_bytes_read +=1
            int_datapoint = int.from_bytes(datapoint, endianness)
            
            if EOF:
                break
            
            datavector.append(int_datapoint)
        file_mrk.close()
        print('Raw marker array: ', datavector)
        data_array= np.array(datavector)
        return(data_array)
    
    def read_dat(self, bytesize = intsize_in_bytes):
        file_dat = open(self.dataname,'rb')
        EOF = False
        datavector = []
        data_bytes_read = 0
        while True:
            datapoint = file_dat.read(bytesize)
            if (len(datapoint) < bytesize):
                print('EOF data file')
                EOF = True
                break
            else:
                data_bytes_read +=1
            int_datapoint = int.from_bytes(datapoint, endianness)
            
            if EOF:
                break
            
            datavector.append(int_datapoint)
        file_dat.close()
        
        return(datavector)
            
        
    # This is just going to read the entire file as bytes until end of file and arrange in an array
    # Skips the first header_size bytes (defaults to 0)
    # Data set is of size intsize_in_bytes and there are num_leads in each time vector
    def read_vectors(self, num_leads = num_leads):
        # Open file and offset the header
        file = open(self.filename, 'rb')
        file.seek(header_size, 0)
        EOF=False
        bytes_read = 0
        
        while True:
            leads_vector_datapoint=[]
        
            # Read num_lead points
            for leadno in range(num_leads):
                point = file.read(intsize_in_bytes)
                if (len(point) < intsize_in_bytes):
                    print('EOF reached of main ECG file.  Total datapoints read = ', bytes_read)
                    EOF = True
                    break
                else:
                    bytes_read += 1
                    
                # Convert to integer and append
                int_point = int.from_bytes(point, endianness)
            
                leads_vector_datapoint.append(int_point)
            
            if EOF:
                #print('EOF with final leads_vector')
                break
        
            yield(leads_vector_datapoint)

        file.close()

    # Return a num points x num_leads array of all the data
    def ecg_array(self):
        ecg_list=[]
        for vector in self.read_vectors(num_leads):
            ecg_list.append(vector)
        ecg_array = np.vstack(ecg_list)
        return(ecg_array)
    
    def split_arrays(self, ecg_array, marker_array, split_divisor=split_divisor_default):
        print('Using marker array: ', marker_array,'\n')
        splits = []
        
        # Make each marker_array into a separate array
        for x in range(len(marker_array)):
            # If first point then split from start of the array
            if x == 0:
                xdiv = int(marker_array[x] / split_divisor)
                splits.append(ecg_array[0:(xdiv-1)])
            
            # Make sure not the last split - in case split to the end
            if x == (len(marker_array)-1):
                xdiv = int(marker_array[x] / split_divisor)
                splits.append(ecg_array[xdiv:])
            
            # Split the array between current marker and previous marker
            else:
                xdiv = int(marker_array[x] / split_divisor)
                prevxdiv = int(marker_array[x-1] / split_divisor)
                splits.append(ecg_array[prevxdiv:xdiv-1])
        return(splits)

    def debias_leads(self, ecg_array):
        # De-bias each lead individually.  Do not aggregate
        sums=np.sum(ecg_array, axis=0)
        norm_sums = sums/ecg_array.shape[0]
        norm_ecg = ecg_array - norm_sums
        return(norm_ecg)
    
    # Upsample to higher sampling rate using linear interpolation.
    def upsample_ecg(self, ecg_array, new_freq, old_freq=samp_rate):
        resamp_lead_array = []
        upsamp_ratio = new_freq / old_freq
        
        old_length = ecg_array.shape[0]
        new_length = int(old_length * upsamp_ratio)
        print('Upsampling to new length ', new_length)

        new_x = np.arange(0, old_length, old_length/new_length)
        old_x = range(old_length)
        
        for leadno in range(ecg_array.shape[1]):
            ecg_lead = ecg_array[:,leadno]
            resamp_lead = np.interp(new_x, old_x, ecg_lead)
            resamp_lead_array.append(resamp_lead)
        return(np.transpose(np.vstack(resamp_lead_array)))

    def upsample_mrk(self, marker_array, new_freq, old_freq=samp_rate):
        upsamp_ratio = new_freq / old_freq
        resamp_marker_array = marker_array * upsamp_ratio      
        return(resamp_marker_array)

if __name__ == "__main__" and "get_ipython" not in dir():
    parser = argparse.ArgumentParser(
        description='Output CSV file from CONTEC TLC5000 ECG file')
    parser.add_argument("filename")
    parser.add_argument("outputfile")
    parser.add_argument("-s", "--split", action='store_true', help = "Split recordings using marker file")
    parser.add_argument("-d", "--debias", action='store_true', help = "Debias / center recordings")
    parser.add_argument("-r", "--resample", type=int, default = 512, help = "Resample to new sampling rate Hz")
    args = parser.parse_args()

    a=contec_ecg(args.filename)
    ecgarray=a.ecg_array()+global_offset
    markers=a.read_mrk()
    
    s = False
    d = False
    r = False

    try:
        if args.split == True:
            s=True
    except AttributeError:
            s=False

    try:
        if args.debias == True:
            d=True
    except AttributeError:
            d=False
            
    try:
        if args.resample.is_integer():
            r=True
    except AttributeError:
            r=False

    if d == True:
        ecgarray=a.debias_leads(ecgarray)

    if r == True:
        #split_divisor_resamp = args.resample
        # Upsample both the ecg array and indices
        ecgarray=a.upsample_ecg(ecgarray, 512)
        markers=a.upsample_mrk(markers, 512)

    if s == True:
        # If ECG resampled adjust marker values accordingly
        #split_divisor_resamp = split_divisor_default
        
        splits = a.split_arrays(ecgarray, markers, split_divisor_resamp)
        print('Split into ',len(splits), 'files.\n')
        for splitnum in range(len(splits)):
            np.savetxt(args.outputfile+'s'+str(splitnum), splits[splitnum], fmt='%.f',delimiter=",")
    else:
        print("Saving output in file ", args.outputfile,"\n")
        np.savetxt(args.outputfile, ecgarray, fmt='%.f',delimiter=",")
