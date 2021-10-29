# import msgpack file and write it into a txt file
import msgpack
import os
import sys

file_name = sys.argv[1]
#file_name = './reference_locations.msgpack'

print(file_name)

with open(file_name + ".msgpack", "rb") as data_file:
	byte_data = data_file.read()
	
data = msgpack.unpackb(byte_data)
# print(data)

with open(file_name + ".txt", "w") as text_file:
    print(data, file=text_file)