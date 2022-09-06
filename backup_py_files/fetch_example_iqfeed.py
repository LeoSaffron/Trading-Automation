# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 02:15:11 2020

@author: jonsnow
"""

# iqfeed.py

import sys
import socket
import pandas as pd

def read_historical_data_socket(sock, recv_buffer=4096):
    """
    Read the information from the socket, in a buffered
    fashion, receiving only 4096 bytes at a time.

    Parameters:
    sock - The socket object
    recv_buffer - Amount in bytes to receive per read
    """
    buffer = ""
    data = ""
    while True:
        data = sock.recv(recv_buffer)
        buffer += str(data)

        # Check if the end message string arrives
        if "!ENDMSG!" in buffer:
            break
   
    # Remove the end message string
    buffer = buffer[:-12]
    return buffer



# iqfeed.py

if __name__ == "__main__":
    # Define server host, port and symbols to download
    host = "127.0.0.1"  # Localhost
    port = 9100  # Historical data socket port
    syms = ["SPY", "AAPL", "GOOG", "AMZN"]

    # Download each symbol to disk
    for sym in syms:
        print ("Downloading symbol: {}...".format(sym))

        # Construct the message needed by IQFeed to retrieve data
        message = (b"HIT," + bytes(sym,"utf8") + b",60,20140101 075000,,,093000,160000,1\n")

        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send the historical data request
        # message and buffer the data
        sock.sendall(message)
        data = read_historical_data_socket(sock)
        sock.close

        # Remove all the endlines and line-ending
        # comma delimiter from each record
        data = "".join(data.split("\r"))
        data = data.replace("b\'", "")
        data = data.replace("!E", "")
        data = data.replace("\'", "")
#        data = data.replace(",\n","\n")[:-1]
        data = data.split(",\\r\\n")
        df = []
        for line in data:
            line_arr = line.split(',')
            if(len(line_arr) == 7):
                df.append(line.split(','))
            else:
                print(line_arr)
        df = pd.DataFrame(df)
        df.columns = ['Date', 'Open', 'Low', 'High', 'Close', 'Volume', 'Open Interest']
        df = df.set_index('Date')
        
        filename = "%s.csv" % sym
        df.to_csv(filename)
        # Write the data stream to disk
#        f = open("%s.csv" % sym, "w")
#        f.write(data)
#        f.close()