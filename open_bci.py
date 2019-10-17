# coding=utf-8
"""
Core OpenBCI object for handling connections and samples from the board.

EXAMPLE USE:

def handle_sample(sample):
  print(sample.channel_data)

board = OpenBCIBoard()
board.print_register_settings()
board.start_streaming(handle_sample)

NOTE: If daisy modules is enabled, the callback will occur every two samples, hence "packet_id" will only contain even numbers. As a side effect, the sampling rate will be divided by 2.

FIXME: at the moment we can just force daisy mode, do not check that the module is detected.
TODO: enable impedance

"""

import serial
import struct
import numpy as np
import time
import timeit
import atexit
import logging
import threading
import sys
import pdb
import glob

SAMPLE_RATE = 250.0  # Hz
START_BYTE = 0xA0  # Byte que indica el inicio de un paquete de datos
END_BYTE = 0xC0  # Byte que indica el final de un paquete de datos

'''
#Commands for in SDK http://docs.openbci.com/software/01-Open BCI_SDK:

command_stop = "s";
command_startText = "x";
command_startBinary = "b";
command_startBinary_wAux = "n";
command_startBinary_4chan = "v";
command_activateFilters = "F";
command_deactivateFilters = "g";
command_deactivate_channel = {"1", "2", "3", "4", "5", "6", "7", "8"};
command_activate_channel = {"q", "w", "e", "r", "t", "y", "u", "i"};
command_activate_leadoffP_channel = {"!", "@", "#", "$", "%", "^", "&", "*"};  //shift + 1-8
command_deactivate_leadoffP_channel = {"Q", "W", "E", "R", "T", "Y", "U", "I"};   //letters (plus shift) right below 1-8
command_activate_leadoffN_channel = {"A", "S", "D", "F", "G", "H", "J", "K"}; //letters (plus shift) below the letters below 1-8
command_deactivate_leadoffN_channel = {"Z", "X", "C", "V", "B", "N", "M", "<"};   //letters (plus shift) below the letters below the letters below 1-8
command_biasAuto = "`";
command_biasFixed = "~";
'''

class OpenBCIBoard(object):
  def __init__(self, port=None, baud=115200, filter_data=True,
    timeout=None):

    """
    Se encarga de realizar la conexión con la tarjeta OpenBci.
    Define las funciones para desempaquetar los datos, comunicarse
    con la tarjeta, iniciar y detener el envío de datos.

    """

## @brief Indica si está activado ó detenido el envío de datos.
    self.streaming = False

## @brief Velocidad de la conexión serial.
    self.baudrate = baud

## @brief Tiempo máximo para la conexión serial.
    self.timeout = timeout

    if not port:
      port = self.find_port()


## @brief Si no se definió el puerto para la conexión lo busca, de lo contrario se conecta al puerto definido.
    self.port = port

    print("Connecting to V3 at port %s" %(port))
    self.ser = serial.Serial(port= port, baudrate = baud, timeout=timeout)
    print("Serial established...")

    time.sleep(2)
    # inicializa la tarjeta de 32 bits.
    self.ser.write(b'v');

    # espera a que el dispositivo esté listo
    time.sleep(1)
    self.print_incoming_text()

## @brief Activa un filtro interno de la tarjeta, ayuda a la estabilización de los datos.
    self.filtering_data = filter_data

## @brief Número de canales del dispositivo para el EEG
    self.eeg_channels_per_sample = 8 # de la tarjeta

## @brief Número de canales auxiliares del dispositivo
    self.aux_channels_per_sample = 3 # de la tarjeta

    self.read_state = 0
    self.attempt_reconnect = False
    self.last_reconnect = 0
    self.reconnect_freq = 5
    self.packets_dropped = 0

    # Disconnects from board when terminated
    atexit.register(self.disconnect)


##  Función para manejar la recepción de datos del dispositivo.
#   Se le debe ingresar un llamado con el cual manejar los datos de un paquete.
#   Verifica que el llamado sea de forma correcta, lee un paquete de datos de la tarjeta
#   mediante "_read_serial_binary" y se los entrega al llamado.
#   CUIDADO: Los datos que se entregan están sin escalar.
#   El factor de escala para dejar los datos en micro-voltios es: 2.23517444553071e-08.
#   @param  . Requiere el llamado(callback) que manejará el paquete de datos.
##  @retval . Le permite al llamado(callback) recivir los datos del paquete.
  def start_streaming(self, callback):

    # Si no está en streaming inicielo e indíqueselo a la tarjeta.
    if not self.streaming:
      self.ser.write(b'b')
      self.streaming = True

    # Si el llamado viene solo lo encapsula en una lista
    if not isinstance(callback, list):
      callback = [callback]

## @var sample
## @brief Se guarda el resultado de "_read_serial_binary" que es quien desempaqueta los datos que entrega la tarjeta.
    sample = self._read_serial_binary()
    
    # Devuelve los datos en el llamado
    for call in callback:
      call(sample) 
  
  def start_streaming_impedance(self):

    # Si no está en streaming inicielo e indíqueselo a la tarjeta.
      self.ser.write(b'z')
      self.ser.write(b'1')
      self.ser.write(b'1')
      self.ser.write(b'0')
      self.ser.write(b'Z')
      b = self.ser.read(100)
      print(b)
      return b
 


 ##  Se encarga de leer los datos directo de la tarjeta y desempaquetarlos.
#   Estructura en la que vienen los datos desde el OpenBci:
#   Start Byte(1)|Sample ID(1)|Channel Data(24)|Aux Data(6)|End Byte(1)
#   0xA0|0-255|8 canales, c/u enteros de 3-bytes con signo|3 enteros de 2-bytes con signo|0xC0
#   @param  . Como parametro opcional está la cantidad máxima de bytes a obviar antes de considerarse un error e intentar reconectar.
##  @retval . Devuelve un objeto creado por "OpenBCISample" que encapsula la id del paquete, el valor de los canales y los datos auxiliares.
  def _read_serial_binary(self, max_bytes_to_skip=10000):


##  Verifica que se esté leyendo correctamente los datos del puerto serial.
#   Si efectivamente lee datos del puerto serial los devuelve, sino avisa de la situacíón y detiene la ejecución.
#   @param  . Tiene como parametro el numero de bytes a leer del puerto serial.
##  @retval . Si la lectura fue correcta devuelve la cantidad indicada de bytes.
    def read(n):
      b = self.ser.read(n)
      if not b:
        self.warn('Device appears to be stalled. Quitting...')
        sys.exit()
        raise Exception('Device Stalled')
        sys.exit()
        return '\xFF'
      else:
        return b

    # Si no llega a su parámetro de salida, vuelve intentar leer siempre que no se supere la cantidad máxima de bytes a obviar
    for rep in range(max_bytes_to_skip):

      #---------Start Byte & ID---------
      if self.read_state == 0:
        
        b = read(1)
        
        if struct.unpack('B', b)[0] == START_BYTE:
          if(rep != 0):
            self.warn('Skipped %d bytes before start found' %(rep))
            rep = 0;
          packet_id = struct.unpack('B', read(1))[0] # La id del paquete va de 0-255

          self.read_state = 1

      #---------Channel Data---------
      elif self.read_state == 1:
        channel_data = []
        for c in range(self.eeg_channels_per_sample):

          # Enteros de 3 bytes que corresponden a los datos de cada canal
          literal_read = read(3)

          unpacked = struct.unpack('3B', literal_read)

          # Los enteros vienen con signo
          if (unpacked[0] >= 127):
            pre_fix = bytes(bytearray.fromhex('FF')) 
          else:
            pre_fix = bytes(bytearray.fromhex('00'))


          literal_read = pre_fix + literal_read;

          # Desempaqueta por little endian(>) el entero con signo 
          myInt = struct.unpack('>i', literal_read)[0]
## @var myInt
## @brief Se guardan los datos de los canales crudos (Los que entrega el ADC).

          channel_data.append(myInt)

        self.read_state = 2;

      #---------Accelerometer Data---------
      elif self.read_state == 2:
        aux_data = []
        for a in range(self.aux_channels_per_sample):

          # short = h
          acc = struct.unpack('>h', read(2))[0]
## @var acc
## @brief Se guardan los datos auxiliares crudos (Los que entrega el ADC).

          aux_data.append(acc)

        self.read_state = 3;

      #---------End Byte---------
      elif self.read_state == 3:
        val = struct.unpack('B', read(1))[0]
        self.read_state = 0 # Lea el siguiente paquete si no está el end byte.
        if (val == END_BYTE):
##  @var sample
##  @brief El resultado de llamar a "OpenBCISample" con la id del paquete, los datos de los canales y los datos auxiliares. 
          sample = OpenBCISample(packet_id, channel_data, aux_data)
          self.packets_dropped = 0
          return sample
        else:
          self.packets_dropped = self.packets_dropped + 1

  
##  Detiene la conexión.
#   Le dice a la tarjeta que pare de enviar datos, y actualiza el valor de streaming por falso.
#   @param  . No requiere parámetros. 
##  @retval . No devuelve nada.
  def stop(self):
    self.streaming = False
    self.ser.write(b's')


##  Se desconecta de la tarjeta.
#   Si aún está enviando datos la tarjeta, llama a "stop", y se encarga de cerrar el puerto serial.
#   @param  . No requiere parámetros. 
##  @retval . No devuelve nada.
  def disconnect(self):
    if(self.streaming == True):
      self.stop()
    if (self.ser.isOpen()):
      print("Closing Serial...")
      self.ser.close()
      logging.warning('serial closed')

  
##  Muestra en pantalla una señal de cuidado.
#   Muestra el aviso en pantalla de cuidado según el texto que se le ingrese.
#   @param  . Como parámetro está el texto que se desea mostrar.
##  @retval . No devuelve nada.     
  def warn(self, text):
    print("Warning: %s" % text)


##  Muestra en pantalla la información de la tarjeta
#   Cuando se inicia la conexión, muestra los datos de debug hasta que llegue la sequencia de fin '$$$'.
#   @param  . No requiere parámetros.
##  @retval . No devuelve ningún valor.
  def print_incoming_text(self):
    line = ''
    # Espera a que el dispositivo envíe datos
    time.sleep(1)
    
    if self.ser.inWaiting():
      line = ''
      c = ''
     # Busca la secuencia de fin '$$$'.
      while '$$$' not in line:
        c = self.ser.read().decode('utf-8')
        line += c
      print(line);
    else:
      self.warn("No Message")


##  Muestra las configuraciones registradas por la tarjeta
#   Le dice a la tarjeta que envíe la información sobre las configuraciones registradas y la imprime con "print_incoming_text".
#   @param  . No requiere parámetros.
##  @retval . No devuelve ningún valor.
  def print_register_settings(self):
    self.ser.write(b'?')
    time.sleep(0.5)
    self.print_incoming_text();


##  Chequea que la conexión esté bien con la tarjeta
#   Si ha perdido muchos paquetes, trata de reconectarse a la tarjeta.
#   @param  . Se le puede ingresar el intervalo para intentar reconectar y el número máximo de paquetes que indicarían un error para proceder a la reconexión.
##  @retval . No devuelve ningún valor.
  def check_connection(self, interval = 2, max_packets_to_skip=10):
    if self.packets_dropped > max_packets_to_skip:
      self.reconnect()
    threading.Timer(interval, self.check_connection).start()


##  Realiza la reconexión con la tarjeta
#   Detiene la coneción actual e inicia una nueva conexión.
#   @param  . No requiere parámetros.
##  @retval . No devuelve ningún valor.
  def reconnect(self):
    self.packets_dropped = 0
    self.warn('Reconnecting')
    self.stop()
    time.sleep(0.5)
    self.ser.write(b'v')
    time.sleep(0.5)
    self.ser.write(b'b')
    time.sleep(0.5)
    self.streaming = True


##  Imprime los paquetes entrantes
#   Funciona como "_read_serial_binary" pero está diseñada para el Debug, en vez de guardar el paquetes, lo muestra en pantalla.
#   @param  . No requiere parámetros.
##  @retval . No devuelve ningún valor.
  def print_packets_in(self):
    while self.streaming:
      b = struct.unpack('B', self.ser.read())[0];
      
      if b == START_BYTE:
        self.attempt_reconnect = False
        if skipped_str:
          logging.debug('SKIPPED\n' + skipped_str + '\nSKIPPED')
          skipped_str = ''

        packet_str = "%03d"%(b) + '|';
        b = struct.unpack('B', self.ser.read())[0];
        packet_str = packet_str + "%03d"%(b) + '|';
        
        # Canales de datos
        for i in range(24-1):
          b = struct.unpack('B', self.ser.read())[0];
          packet_str = packet_str + '.' + "%03d"%(b);

        b = struct.unpack('B', self.ser.read())[0];
        packet_str = packet_str + '.' + "%03d"%(b) + '|';

        # Canales auxiliares
        for i in range(6-1):
          b = struct.unpack('B', self.ser.read())[0];
          packet_str = packet_str + '.' + "%03d"%(b);
        
        b = struct.unpack('B', self.ser.read())[0];
        packet_str = packet_str + '.' + "%03d"%(b) + '|';

        # Byte final
        b = struct.unpack('B', self.ser.read())[0];
        
        # Paquete válido
        if b == END_BYTE:
          packet_str = packet_str + '.' + "%03d"%(b) + '|VAL';
          print(packet_str)
        
        # Paquete no válido
        else:
          packet_str = packet_str + '.' + "%03d"%(b) + '|INV';
          # Reconectarse
          self.attempt_reconnect = True
                
      else:
        print(b)
        if b == END_BYTE:
          skipped_str = skipped_str + '|END|'
        else:
          skipped_str = skipped_str + "%03d"%(b) + '.'

      if self.attempt_reconnect and (timeit.default_timer()-self.last_reconnect) > self.reconnect_freq:
        self.last_reconnect = timeit.default_timer()
        self.warn('Reconnecting')
        self.reconnect()


##  Activa los filtros internos de la tarjeta
#   le dice a la tarjeta que active los filtros internos que tiene.
#   @param  . No requiere parámetros.
##  @retval . No devuelve ningún valor.
  def enable_filters(self):
    self.ser.write(b'f')
    self.filtering_data = True;


##  Desctiva los filtros internos de la tarjeta
#   le dice a la tarjeta que desactive los filtros internos que tiene.
#   @param  . No requiere parámetros.
##  @retval . No devuelve ningún valor.
  def disable_filters(self):
    self.ser.write(b'g')
    self.filtering_data = False;


##  Encuentra el puerto donde está conectado el openBci
#   Segun el sistema operativo, busca si en alguno de los puertos está conectado el openBci.
#   @param  . No requiere parámetros.
##  @retval . Devuelve el puerto al cual está conectado el openBci.
  def find_port(self):
    print('Searching Board...')
    # nombre de los puertos seriales
    if sys.platform.startswith('win'):
      ports = ['COM%s' % (i+1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
      ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
      ports = glob.glob('/dev/tty.usbserial*')
    else:
      raise EnvironmentError('Error finding ports on your operating system')
    openbci_port = ''
    for port in ports:
      try:
        s = serial.Serial(port= port, baudrate = self.baudrate, timeout=self.timeout)
        s.write(b'v')
        openbci_serial = self.openbci_id(s)
        s.close()
        if openbci_serial:
          openbci_port = port;
      except (OSError, serial.SerialException):
        pass
    if openbci_port == '':
      raise OSError('Cannot find OpenBCI port')
    else:
      return openbci_port

##  Busca id del OpenBci en un puerto dado
#   Está enfocada para la conexión automatica al dispositivo, preguntando en el puerrto indicado si tiene como respuesta la id del openBci. 
#   @param  . Requiere el puerto serial sobre el cual se va a buscar.
##  @retval . Devuelve verdadero si el puerto indicado tiene la id del OpenBci, falso de lo contrario.
  def openbci_id(self, serial):
  	line = ''
  	# Espera a que el dispositivo envíe datos
  	time.sleep(2)

  	if serial.inWaiting():
  	  line = ''
  	  c = ''
  	 # Busca la secuencia de fin '$$$'
  	  while '$$$' not in line:
  	    c = serial.read().decode('utf-8')
  	    line += c
  	  if "OpenBCI" in line:
  	    return True
  	return False

class OpenBCISample(object):
  def __init__(self, packet_id, channel_data, aux_data):

    """
    Objeto encargado de encapsular cada una de las muestras de la tarjeta OpenBci.
    Encapsula los parámetros de entrada "packet_id", "channel_data" y "aux_data" para
    entregarselos al llamado en "_read_serial_binary" de la clase "OpenBCIBoard".

    """
## @brief La id del paquete en cuestión.  
    self.id = packet_id;

## @brief Los datos de los canales para el paquete que se está encapsulando.
    self.channel_data = channel_data;

## @brief Los datos auxiliares para el paquete que se está encapsulando.
    self.aux_data = aux_data;


