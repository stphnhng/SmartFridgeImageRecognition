#!/usr/bin/env python

import socket


TCP_IP = '172.20.10.10'
TCP_PORT = 8888
BUFFER_SIZE = 20  # Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print 'Connection address:', addr
num = 0
while True:
   
    #if not data: break
    print "received data:", num
    conn.send(str(num))  # echoi
    num = num +1 
#conn.close()
