import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from optparse import OptionParser

def readCommand(argv):
	usage = ""
	parser = OptionParser()
	parser.add_option("-f", "--input", dest="input", action="store")
	parser.add_option("-s", "--output", dest="output", action="store")
	parser.add_option("-m", "--nrow", dest="nrow", type="int", action="store")
	parser.add_option("-n", "--ncol", dest="ncol", type="int", action="store")
	options, junks = parser.parse_args(argv)
	if len(junks) != 0:
		raise Exception("Cannot understand: " + str(junks))
	args = dict()
	args["ifilename"] = options.input
	args["ofilename"] = options.output
	args["nrow"] = options.nrow
	args["ncol"] = options.ncol
	if not options.nrow:
		args["nrow"] = options.ncol
	elif not options.ncol:
		args["ncol"] = options.nrow
	return args

def get_data_binary(ifilename, nrow, ncol):

	''' Plot double data in binary format '''

	data = np.fromfile(ifilename)
	data = [data[i*ncol:(i+1)*ncol] for i in range(nrow)]
	return data

def get_data_xy(ifilename, nrow, ncol):

	''' Plot double data in txt format '''

	data = np.loadtxt(ifilename, skiprows=0)
	data = [*zip(*data)]
	return data

if __name__ == '__main__':
	args = readCommand(sys.argv[1:])
	data = get_data_xy(args["ifilename"], args["nrow"], args["ncol"])
	#data = get_data_binary(args["ifilename"], args["nrow"], args["ncol"])
	plt.cla()
	plt.imshow(data, cmap='gray')
	if args["ofilename"]:
		plt.savefig(args["ofilename"])
	else:
		plt.show()

"""
python3 plot2D.py -[fsmn] [--input|--output|--nrow|--ncol]
"""