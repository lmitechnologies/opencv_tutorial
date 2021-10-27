#demo.py - get opt module
__author__ = 'nixCraft'
import sys,getopt #get argument list using sys module

#Store input and output file names
ifile=''
ofile=''

#Read command line args

myopts,args = getopt.getopt(sys.argv[1:],"i:o:")

#o == option
#a == argument passed to o


for o,a in myopts:
    if o =='-i':
        ifile=a
    elif o == '-o':
        ofile=a
    else:
        print("Usage: %s -i input -o output" %sys.argv[0])


#Display input and output files passed as args
print("Input file: %s and output file: %s" %(ifile,ofile))

