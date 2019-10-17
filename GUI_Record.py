# coding=utf-8


from PyQt4 import QtGui, QtCore
# import pyqtgraph as pg
import random 
import numpy as np


# Módulos
import sys; sys.path.append('..') 
import open_bci as bci 
import os
import logging
import time as tm
from datetime import datetime, date
#import pyedflib
# import numpy as np
from numpy.matlib import repmat
from scipy.signal import lfilter
import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.ptime import time
from scipy.signal import filtfilt, parzen
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
import cPickle
#from matplotlib import pyplot as mplt


##  Llamado para recibir los datos.
#   Es el llamado necesario en "start_streaming" del módulo "open_bci_GCPDS" para realizar el manejo de los datos
#   que llegan de la tarjeta, en este caso los agrega en la variable eeg.
#   @param  . cuando se llama de la manera "board.start_streaming(saveData)" no requiere parametros .
##  @retval . No devuelve nada. pero sí agrega los valores de los canales a la variable eeg.
def saveData(sample):
	global eeg
	eeg.append(sample.channel_data)


##  Conexión con la tarjeta openBci
#   Crea el objeto board mediante "OpenBCIBoard" del módulo "open_bci_GCPDS" para manejar el envío de datos de la tarjeta.
#   @param  . No requiere parámetros. 
##  @retval . Devuelve el objeto "Board" con las propiedades de la clase "openBCIBoard".
def connect_board():
	baud = 115200
	board = bci.OpenBCIBoard(port=None, baud = baud, filter_data = True)
	print("Board Connected")
	return board


##  Inizializa la tarjeta
#   Le dice a la tarjeta que inicie el envío de datos, activa los filtros de la misma, define las variables para los filtros
#   predeterminados que son: Notch en 60 hz, pasa bandas orden 3 Butterworth de 5 - 45 hz.
#   @param  . Requiere el objeto board entregado por "connect_board". 
##  @retval . No devuelve nada.
def initialize(board):
	global b, a, eeg, b_n, a_n, filter_used, filter_notch_used, eeg_to_save
	eeg = []	
	board.ser.write('v')	
	tm.sleep(1)
	board.enable_filters()	

# 	b_n = [1.0000,   -0.1297,    1.0000]	# notch: 60 hz.
# 	a_n = [1.0000,   -0.1032,    0.5914]
# 	b = [0.0579,         0,   -0.1737,         0,    0.1737,         0,   -0.0579]		# 5 - 45Hz butter filter
# 	a = [1.0000,   -3.7335,    5.9137,   -5.2755,    2.8827,   -0.9042,    0.1180]
# 	filter_used = 'PB: 5-45Hz'
# 	filter_notch_used = 'N: 60Hz'

	tm.sleep(0.1)		
	board.start_streaming(saveData)
	print('Board initializated')


##  Se desconecta de la tarjeta.
#   Graba una última muestra pero no la guarda, procede directamente a desconectarse de la tarjeta.
#   @param  . Requiere el objeto board entregado por "connect_board".  
##  @retval . No devuelve nada.
def disconnect_board(board):
	global eeg
	eeg = []
	board.ser.write('v')
	tm.sleep(0.1)
	board.start_streaming(saveData)
	print('Streaming ended')
	print('')
	board.disconnect()
	sys.exit()


##  Pre proceso del EEG.
#   Como los datos de la tarjeta son crudos, se escalan adecuadamente para que estén en uVoltios.
#   Luego procede a centrarlos en la media y realizar el filtrado segun [b,a] --> pasabandas, y [b_n,a_n] --> notch.
#   Los datos crudos se escalan multiplicando por 2.23517444553071e-08.
#   @param  . Requiere la lista "eeg", que es entregada por "get_n_secs" (EEG desde la tarjeta, cada canal en una columna). 
##  @retval . Devuelve "eeg_processed" que tiene las mismas dimensiones de la matriz "eeg" de entrada, pero está transpuesta.
def pre_process(eeg):
	eeg = np.array(eeg)
	[fil,col] = eeg.shape	
	eeg_processed = np.zeros([fil,col])
	for i in range(fil):
		data = eeg[i,:] * 2.23517444553071e-08	# 2.23517444553071e-08 factor de escala en uV
		data = data - np.mean(data)
		data = lfilter(b_n,a_n, data)
		data = lfilter(b,a,data)
		eeg_processed[i,:] = data
	return(eeg_processed)


##  Obtiene "n" segundos de datos de la tarjeta openBci.
#   Llama a la función "start_streaming" del módulo "open_bci_GCPDS" que entrega un paquete de datos, hasta que se cumpla
#   el número de muestras correspondientes a "n" segundos (la frecuencia de muestreo es 250). 
#   @param  . Como parametros requiere el objeto board entregado por "connect_board" y la cantidad de segundos "n" que se desea ontener.    
##  @retval . Devuelve una matriz de el número de datos en el tiempo "n" por canales.

app = QtGui.QApplication([])
pg.setConfigOption('background', 'w')

class MainWindow(QtGui.QMainWindow):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
    def __init__(self):
#         super(Window, self).__init__()
        QtGui.QMainWindow.__init__(self) # inicializa ventana ppal
    
        self.screenShape = QtGui.QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, self.screenShape.width(), self.screenShape.height())
        self.setStyleSheet("QMainWindow {background: 'white';}")
        self.setWindowTitle("OpenBCI_GUI")
        app_icon = QtGui.QIcon()
        app_icon.addFile('/home/sebastian_gil/Tesis/codes_open_bci_8_channels/cog_1024x1024.png', QtCore.QSize(1024,1024))
        self.setWindowIcon(app_icon)
#         self.setWindowIcon(QtGui.QIcon('cog_1024x1024.png'))

        self.pic = QtGui.QLabel(self)
        self.pic.setPixmap(QtGui.QPixmap("icon.png"))
        self.pic.setGeometry(0, 0, self.screenShape.width(), 512)  
        self.pic.setAlignment(QtCore.Qt.AlignCenter)
        self.pic.show() 
    
        self.name = QtGui.QLabel("NEUROCONVI - RECORD",self)
        newfont = QtGui.QFont("Helvetica", 20, QtGui.QFont.Bold) 
        self.name.setFont(newfont)
        self.name.resize(self.screenShape.width(),120)
        self.name.move(0,500)
        self.name.setAlignment(QtCore.Qt.AlignCenter)

        self.btn = QtGui.QPushButton("StartSystem", self)
        self.btn.resize(120,120)
        self.btn.move((self.screenShape.width()/2)-60,600)       
        self.btn.clicked.connect(self.home)
        self.show()
        
#--------------------------BANDPASS 1-50-------------------------------------------------------------
#         self.b = [ 0.200138725658073, 0, -0.400277451316145, 0, 0.200138725658073 ]
#         self.a = [ 1, -2.35593463113158, 1.94125708865521, -0.784706375533419, 0.199907605296834 ]
        self.b = [0.0579,         0,   -0.1737,         0,    0.1737,         0,   -0.0579]		# 5 - 45Hz butter filter
        self.a = [1.0000,   -3.7335,    5.9137,   -5.2755,    2.8827,   -0.9042,    0.1180]
#         self.b_n = [1.0000,   -0.1297,    1.0000]	# notch: 60 hz.
#         self.a_n = [1.0000,   -0.1032,    0.5914]
        self.b_n = [0.965080986344733, -0.242468320175764, 1.94539149412878, -0.242468320175764, 0.965080986344733]
        self.a_n = [1, -0.246778261129785, 1.94417178469135, -0.238158379221743, 0.931381682126902]
        
        self.keyPressed.connect(self.on_key)
#         self.home()
    
    def keyPressEvent(self, event):
        super(MainWindow, self).keyPressEvent(event)
        self.keyPressed.emit(event)
        
    def get_n_secs(self,board,n):
        global eeg
        eeg = []

        for i in range(int(round(n*250))):
            board.start_streaming(saveData)
            self.counter+=self.Ts

        return(eeg)
    
    def pre_process_plot(self,eeg):
        eeg = np.array(eeg)
        [fil,col] = eeg.shape	
        eeg_processed = np.zeros([fil,col])
        for i in range(fil):
            data = eeg[i,:] * 2.23517444553071e-02	# 2.23517444553071e-08 factor de escala en uV
            data = data - np.mean(data)
            data = lfilter(self.b_n,self.a_n, data)
            data = lfilter(self.b,self.a,data)       
            eeg_processed[i,:] = data
        return(eeg_processed)
    
    def home(self):
       
        self.pic.hide() 
        self.name.hide()
        self.btn.hide()
        self.hide() 
        
        main_widget = QtGui.QWidget(self)
        main_widget.setGeometry(0, 20, self.screenShape.width(), self.screenShape.height()-20)
                
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('EEG Data System')
        #win.resize(1000,1000)
        pg.setConfigOptions(antialias=True)

#--------------------PARAMETERS----------------------------------------------        
        
        self.stim=[] #guarda estimulos
        self.time = 0.4
        self.Ts = 4 #  periodo de muestreo escalado. necesario para evitar errorres (e.g 0.4*3=1.2000000000000002)
        self.fs =  250
        self.counter=-4.0
        self.channels = 8
        self.win_size = 4
        self.datetime = ""

#---------------GRAFICAS-----------------------------------------------------

        #EEG Plot
        self.plt8 = win.addPlot(title="O2",row=7, col=0, colspan=2)
        self.plt8.setWindowTitle('O2')
        self.plt8.setYRange(-50, 50)
        self.plt8.setXRange(0, self.win_size-1)
#         self.plt8.enableAutoRange('y', True)

        self.plt7 = win.addPlot(title="O1",row=6, col=0, colspan=2)
        self.plt7.setWindowTitle('O1')
        self.plt7.setYRange(-50, 50)
        self.plt7.setXRange(0, self.win_size-1)
        self.plt7.hideAxis('bottom')
#         self.plt7.enableAutoRange('y', True)

        self.plt6 = win.addPlot(title="P8",row=5, col=0, colspan=2)
        self.plt6.setWindowTitle('P8')
        self.plt6.setYRange(-50, 50)
        self.plt6.setXRange(0, self.win_size-1)
        self.plt6.hideAxis('bottom')
#         self.plt6.enableAutoRange('y', True)

        self.plt5 = win.addPlot(title="P7",row=4, col=0, colspan=2)
        self.plt5.setWindowTitle('P7')
        self.plt5.setYRange(-50, 50)
        self.plt5.setXRange(0, self.win_size-1)
        self.plt5.hideAxis('bottom')
#         self.plt5.enableAutoRange('y', True)

        self.plt4 = win.addPlot(title="F4",row=3, col=0, colspan=2)
        self.plt4.setWindowTitle('C4')
        self.plt4.setYRange(-50, 50)
        self.plt4.setXRange(0, self.win_size-1)
        self.plt4.hideAxis('bottom')
#         self.plt4.enableAutoRange('y', True)

        self.plt3 = win.addPlot(title="F3",row=2, col=0, colspan=2)
        self.plt3.setWindowTitle('C3')
        self.plt3.setYRange(-50, 50)
        self.plt3.setXRange(0, self.win_size-1)
        self.plt3.hideAxis('bottom')
#         self.plt3.enableAutoRange('y', True)

        self.plt2 = win.addPlot(title="Fp2",row=1, col=0, colspan=2)
        self.plt2.setWindowTitle('FP2')
        self.plt2.setYRange(-50,50)
        self.plt2.setXRange(0, self.win_size-1)
        self.plt2.hideAxis('bottom')
#         self.plt2.enableAutoRange('y', True)

        self.plt1 = win.addPlot(title="Fp1",row=0, col=0, colspan=2)
        self.plt1.setWindowTitle('FP1')
        self.plt1.setLabel('bottom', 'Time', units='sec')
        self.plt1.setYRange(-50, 50)
        self.plt1.setXRange(0, self.win_size-1)
        self.plt1.hideAxis('bottom')
#         self.plt1.enableAutoRange('y', True)

#----------------------BOTONES------------------------------------------------
        
        self.btnStart = QtGui.QPushButton("StartDataStream", self)
        self.btnStart.move(0,0)
        self.btnStart.resize(120,25)
        self.btnStart.clicked.connect(self.Start)
        self.stream = False
        
        self.btnFilter = QtGui.QPushButton("BP:5-45",self)
        self.btnFilter.move(120,0)
        self.btnFilter.resize(self.btnFilter.minimumSizeHint())
        self.btnFilter.clicked.connect(self.Filter)
        self.filt=1
        
        btnScaleVert = QtGui.QComboBox(self)
        btnScaleVert.move(200,0)
        btnScaleVert.resize(self.btnFilter.minimumSizeHint())
        btnScaleVert.addItem("50uV")
        btnScaleVert.addItem("100uV")
        btnScaleVert.addItem("200uV")
        btnScaleVert.addItem("400uV") 
        btnScaleVert.activated[str].connect(self.ScaleVert)
                
#------------------------------LABELS----------------------------------------
        
        self.LblScaleVert = QtGui.QLabel("VertScale", self)
        self.LblScaleVert.move(200,20)
        self.lblFilter = QtGui.QLabel("Filter", self)
        self.lblFilter.move(120,20)
                
#-------------------FIRST RECORD---------------------------------------------

        self.board = connect_board()
        initialize(self.board)
        
        eeg = self.get_n_secs(self.board,self.time)
        eeg = np.asarray(eeg)
        y1 = np.transpose(eeg.tolist())

        y_p1 = self.pre_process_plot(y1)

        x1 = np.linspace(0,self.time,self.fs*self.time)
        
        self.curve = []
        c = self.plt1.plot(x1, y_p1[0], pen='w')
        self.curve.append(c)
        c = self.plt2.plot(x1, y_p1[1], pen='w')  
        self.curve.append(c)
        c = self.plt3.plot(x1, y_p1[2], pen='w')  
        self.curve.append(c)
        c = self.plt4.plot(x1, y_p1[3], pen='w')  
        self.curve.append(c)
        c = self.plt5.plot(x1, y_p1[4], pen='w') 
        self.curve.append(c)
        c = self.plt6.plot(x1, y_p1[5], pen='w')  
        self.curve.append(c)
        c = self.plt7.plot(x1, y_p1[6], pen='w') 
        self.curve.append(c)
        c = self.plt8.plot(x1, y_p1[7], pen='w') 
        self.curve.append(c)
        
        self.x = np.linspace(0,self.win_size,self.fs*self.win_size)
        self.xT = np.linspace(0,self.win_size,self.fs*self.win_size)
        self.y = y1
        
        box_layout = QtGui.QVBoxLayout(main_widget) # widget principal (caja)
        box_layout.addWidget(win)
        self.show()        
        
        
    def ScaleVert(self, text):
        
        if (text=="50uV"):
            self.plt1.setYRange(-50, 50)
            self.plt2.setYRange(-50, 50)
            self.plt3.setYRange(-50, 50)
            self.plt4.setYRange(-50, 50)
            self.plt5.setYRange(-50, 50)
            self.plt6.setYRange(-50, 50)
            self.plt7.setYRange(-50, 50)
            self.plt8.setYRange(-50, 50)
        
        elif (text=="100uV"):
            self.plt1.setYRange(-100, 100)
            self.plt2.setYRange(-100, 100)
            self.plt3.setYRange(-100, 100)
            self.plt4.setYRange(-100, 100)
            self.plt5.setYRange(-100, 100)
            self.plt6.setYRange(-100, 100)
            self.plt7.setYRange(-100, 100)
            self.plt8.setYRange(-100, 100)
            
        elif (text=="200uV"):
            self.plt1.setYRange(-200, 200)
            self.plt2.setYRange(-200, 200)
            self.plt3.setYRange(-200, 200)
            self.plt4.setYRange(-200, 200)
            self.plt5.setYRange(-200, 200)
            self.plt6.setYRange(-200, 200)
            self.plt7.setYRange(-200, 200)
            self.plt8.setYRange(-200, 200)
        else: 
            self.plt1.setYRange(-400, 400)
            self.plt2.setYRange(-400, 400)
            self.plt3.setYRange(-400, 400)
            self.plt4.setYRange(-400, 400)
            self.plt5.setYRange(-400, 400)
            self.plt6.setYRange(-400, 400)
            self.plt7.setYRange(-400, 400)
            self.plt8.setYRange(-400, 400)
            
        
    def Filter(self):
        
        self.filt+=1
        
        if(self.filt>4): 
            self.filt=0
        else: 
            pass
        
        if(self.filt==0):
            self.b = [0.0579,         0,   -0.1737,         0,    0.1737,         0,   -0.0579]
            self.a = [1.0000,   -3.7335,    5.9137,   -5.2755,    2.8827,   -0.9042,    0.1180]
            self.btnFilter.setText("BP:5-45")     
        
        elif(self.filt==1):
            self.b = [ 0.200138725658073, 0, -0.400277451316145, 0, 0.200138725658073 ]
            self.a = [ 1, -2.35593463113158, 1.94125708865521, -0.784706375533419, 0.199907605296834 ]
            self.btnFilter.setText("BP:1-50")         
            
        elif(self.filt==2):
            self.b = [ 0.00512926836610803, 0, -0.0102585367322161, 0, 0.00512926836610803 ]
            self.a = [ 1, -3.67889546976404, 5.17970041352212, -3.30580189001670, 0.807949591420914 ]
            self.btnFilter.setText("BP:7-13")
            
        elif(self.filt==3):    
            self.b = [ 0.117351036724609, 0, -0.234702073449219, 0, 0.117351036724609  ]
            self.a = [ 1, -2.13743018017206, 2.03857800810852, -1.07014439920093, 0.294636527587914 ] 
            self.btnFilter.setText("BP:15-50")
            print(self.filt)
            
        else:
            self.b = [ 0.175087643672101, 0, -0.350175287344202, 0, 0.175087643672101  ]
            self.a = [ 1, -2.29905535603850, 1.96749775998445, -0.874805556449481, 0.219653983913695 ] 
            self.btnFilter.setText("BP:5-50")
           
    def Start(self):
        
        self.stream = not self.stream

        if (self.stream):
                self.btnStart.setText("StopDataStream")
                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self.update)
                self.timer.start(0)
                self.datetime=datetime.now().strftime('Date:%Y-%m-%d_Time:%H:%M:%S')
        else: 
            self.btnStart.setText("StartDataStream")
            self.timer.stop()
            
            filename=raw_input('Guardar como (nombre sin extension): \n(ingresar exit para salir sin guardar) ')
            
            if (filename!="exit"):
                aux=np.zeros((2,self.raw_data.shape[1]))
                self.raw_data=np.r_[self.raw_data,aux] # agregar dos filas de ceros, una para el tiempo y otra para stimulos

                print("Almacenando datos...")

                for i in range (self.raw_data.shape[1]):# llenando el vector de tiempo
                    self.raw_data[-2][i]=(i*4.0)/1000.0

                for i in range(len(self.stim)):
                    self.raw_data[-1][int((self.stim[i]/4)*1000)]=1 # llenando vector de stimulos
            
                np.savetxt('SavedData/'+filename+'.csv',self.raw_data,header=self.datetime+"\n"+"Channels:Fp1,FP2,F3,F4,P7,P8,O1,O2",comments='')
                print("Datos almacenados en: "+os.getcwd()+"/SavedData/"+filename+'.csv')
                print(self.stim)
            
            else:
                print("Datos NO Almacenados")

            eeg = self.get_n_secs(self.board,self.time)
            eeg = np.asarray(eeg)
            self.y = np.transpose(eeg.tolist())
            self.counter=-4.0
            self.stim=[]
       
    
    def update(self):
        global y_p

        EEG_new = self.get_n_secs(self.board,self.time)
        EEG_new = np.asarray(EEG_new)
        y2 = np.transpose(EEG_new.tolist())

        #--------------------------------------------------------
        #EEG Plot
        if np.ceil(self.counter/1000) < self.win_size:
            self.y = np.c_[self.y,y2]
            self.raw_data = self.y
            self.x = self.xT[:self.y.shape[1]]
            y_f = self.pre_process_plot(self.y)
            y_p = y_f
        else:
            self.raw_data = np.c_[self.raw_data,y2]
            self.y = np.c_[self.y,y2]
            self.y = self.y[:,int(self.time*self.fs):]
            self.x = self.xT[:self.y.shape[1]]
            y_f = self.pre_process_plot(self.y)
            y_p = np.c_[y_p,y_f[:,(y_f.shape[1]-int(self.time*self.fs)):y_f.shape[1]]]
            y_p = y_p[:,int(self.time*self.fs):]

        for i in range(self.channels):
            self.curve[i].setData(self.x,y_p[i], pen='k') 
            
    def on_key(self, event):
#         if event.key() == QtCore.Qt.Key_Space:
#             print "enter"  # this is called whenever the continue button is pressed
        if event.key() == QtCore.Qt.Key_S:
#             self.timer.stop()
            self.stim.append(self.counter/1000)

               
if __name__ == '__main__':
	import sys
	app_window = MainWindow()
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()
	disconnect_board(app_window.board)

