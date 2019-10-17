from pyqtgraph.Qt import *
import numpy as np
import pyqtgraph as pg
#import serial
import struct
import time
import timeit
import atexit
import logging
import threading
import sys
import pdb
import glob
import math

n_elec = 8    

class HeadPlot(object):
    
    def __init__(self):
   
        self.rel_width = 0      #Width 
        self.rel_height = 0     #Height
        self.circ_x = 0         #Position in x axe to draw head
        self.circ_y = 0         #Position in y axe to draw head11
        self.circ_diam = 0      #Head Diameter
        self.earL_x = 0         #Position in x axe to draw left ear
        self.earL_y = 0         #Position in y axe to draw left ear
        self.earR_x = 0         #Position in x axe to draw right ear
        self.earR_y = 0         #Position in y axe to draw right ear 
        self.ear_width = 0      #Ear width
        self.ear_height = 0     #Ear Height
        self.nose_x = 0         #Vector Position in x axe to draw nose
        self.nose_y = 0         #Vector Position in y axe to draw nose
        self.electrode_xy = 0   #Electrodes positions
        self.elec_diam = 0      #Electrodes diameter
        self.image_x = 0        #Ref axe x to location electrodes, pixels and image
        self.image_y = 0        #Ref axe y to location electrodes, pixels and image
   

    def Head(self,w,h,win_x,win_y):
        
        self.nose_x = [0,0,0]
        self.nose_y = [0,0,0]
        self.electrode_xy = np.zeros((n_elec,2))
        self.ref_electrode_xy = [0,0]
        self.rel_width = w
        self.rel_height = h
        self.setWindowDimensions(win_x,win_y)
    
    def setWindowDimensions(self,win_width,win_height):
        nose_relLen = 0.075
        nose_relWidth = 0.05
        nose_relGutter = 0.02
        ear_relLen = 0.15
        ear_relWidth = 0.075
        square_width = min(self.rel_width*win_width,self.rel_height*win_height) 
        total_width = square_width 
        total_height = square_width 
        nose_width = total_width * nose_relWidth 
        nose_height = total_height * nose_relLen 
        self.ear_width = int(ear_relWidth * total_width)
        self.ear_height = int(ear_relLen * total_height)
        circ_width_foo = int((total_width - 2.*(self.ear_width)/2.0)) 

        circ_height_foo = int((total_height - nose_height)) 
        self.circ_diam = min(circ_width_foo, circ_height_foo) 
        self.circ_x = int(total_width/2)
        self.circ_y = int((total_height/2) + nose_height)

        self.earL_x = int(self.circ_x - self.circ_diam/2) 
        self.earR_x = int(self.circ_x + self.circ_diam/2)
        self.earL_y = int(self.circ_y)
        self.earR_y = int(self.circ_y)

        self.nose_x[0] = self.circ_x - int(((nose_width/10.)*win_width))
        self.nose_x[1] = self.circ_x + int(((nose_width/10.)*win_width))
        self.nose_x[2] = self.circ_x 
        self.nose_y[0] = self.circ_y - int((self.circ_diam/2.0 - nose_relGutter*win_height))
        self.nose_y[1] = self.nose_y[0]
        self.nose_y[2] = self.circ_y - int((self.circ_diam/2.0 + nose_height))

        elec_relDiam = 0.05 
        self.elec_diam = int(elec_relDiam*(self.circ_diam))
        self.setElectrodeLocations(elec_relDiam)

        self.image_x = int(round(self.circ_x - 0.5*self.circ_diam - 0.5*self.ear_width));
        self.image_y = self.nose_y[2]; 
          
    def setElectrodeLocations(self,elec_realDiam):
        n_elec_to_load = n_elec+1;  
        default_fname = 'sixteenchannels.txt'
        elec_relXY = np.loadtxt(default_fname , delimiter=',',skiprows=1)

        if len(elec_relXY) < n_elec_to_load:
            print("headPlot: electrode position file not found or was wrong size ")

        for i in range(min(len(self.electrode_xy),len(elec_relXY))):
            self.electrode_xy[i][0] = self.circ_x+int((elec_relXY[i][0]*(self.circ_diam)))
            self.electrode_xy[i][1] = self.circ_y+int((elec_relXY[i][1]*(self.circ_diam)))