#demo.py - get opt module
__author__ = 'nixCraft'
import sys,getopt #get argument list using sys module

#Store input and output file names
ifile=''
ofile=''

#Read command line args

myopts,args = getopt.getopt(sys.argv[1:,"i:o:"]

"""
o == option
a == argument passed to o"""

for o,a in myopts:
    if o =='-i':
        ifile=a
    elif o == '-':
        ofile=a
    else:
        print("Usage: %s -i input -o output" %sys.argv[0])


#Display input and output files passed as args
print("Input file: %s and output file: %s" %(ifile,ofile))



#Get total number of arguments passed to demo.py
total = len(sys.argv)

#Get the argument list details
cmdargs = str(sys.argv)

#Printing
print("Tht total number of args passed to the script :%d " %total)
print("Argslist %s" %cmdargs)

#Parsing arguments one by one
"""
print("Script name: %s" %str(sys.argv[0]))
print("First argument name: %s" %str(sys.argv[1]))
print("Second argument name: %s" %str(sys.argv[2]))
"""
print("Script name: %s" %str(sys.argv[0]))
for i in range(total):
    print("Argument %d is %s" %(i,str(sys.argv[i])))
