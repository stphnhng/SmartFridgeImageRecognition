# Import required Python libraries
# -----------------------
import time
import RPi.GPIO as GPIO
import web
import socket

# -----------------------
# Global variables for socket
# -----------------------

TCP_IP = '172.20.10.10'
TCP_PORT = 8888
BUFFER_SIZE = 20  # Normally 1024, but we want fast response

# -----------------------
# Define some functions
# -----------------------

def measure():
      # This function measures a distance

    GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, GPIO.LOW)
    start = time.time()

    while GPIO.input(GPIO_ECHO)==0:
        start = time.time()
        #print(0)
    while GPIO.input(GPIO_ECHO)==1:
        stop = time.time()
        #print(1)
    elapsed = stop - start
    distance = (elapsed * 34300)/2
   #print(distance)
    return distance

def measure_average():
# This function takes 3 measurements and
# returns the average.

    distance1=measure()
    time.sleep(0.1)
    distance2=measure()
    time.sleep(0.1)
    distance3=measure()
    distance = distance1 + distance2 + distance3
    distance = distance / 3
   #print(distance)
    return distance

# -----------------------
# Main Script
# -----------------------

# Use BCM GPIO references
# instead of physical pin numbers
GPIO.setmode(GPIO.BOARD)

# Define GPIO to use on Pi
GPIO_TRIGGER = 23
GPIO_ECHO    = 24

print "Ultrasonic Measurement"

# Set pins as output and input
GPIO.setup(GPIO_TRIGGER,GPIO.OUT,initial = GPIO.LOW)  # Trigger
GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo

# Set trigger to False (Low)
GPIO.output(GPIO_TRIGGER, False)

distance = measure_average()

# -----------------------
# Socket commands
# -----------------------

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print 'Connection address:', addr
num = 0

# -----------------------
# Sensor - Socket Execution
# -----------------------

try:
    while True:
    #print('hi')
        distance = measure_average()
        print "Distance : %.1f" % distance
        if (distance > 100):
            conn.send("Start video capture") 
            print "open"
        else:
            conn.send("Stop video capture")
            print "closed"
        time.sleep(1)

except KeyboardInterrupt:
# User pressed CTRL-C
# Reset GPIO settings
    GPIO.cleanup()
