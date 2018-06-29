import socket
import sys
import gensim
from gensim.models import KeyedVectors
import json
import numpy as np

HOST = 'localhost'   # Symbolic name, meaning all available interfaces
PORT = 1234 # Arbitrary non-privileged port
buf = 1000000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
 
#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
     
print('Socket bind complete')

print('Loading embeddings')
embeddings_model = KeyedVectors.load_word2vec_format('embeddings.bin', binary=True, unicode_errors='ignore')

#Start listening on socket
s.listen(10)
print('Socket now listening')

#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    word=conn.recv(buf).decode()    
    try:
        conn.send(json.dumps(embeddings_model[word].tolist()).encode())
    except:
        conn.send(json.dumps([]).encode())

s.close()
