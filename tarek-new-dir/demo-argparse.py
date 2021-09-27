import argparse

ap=argparse.ArgumentParser(description='This is a demoscript by tfa21.')
ap.add_argument('-i','--input',help='Input file name',required=True)
ap.add_argument('-o','--output',help='Output file name',required=True)
args=ap.parse_args()

##Showing values ##

print("Input file: %s" %args.input)
print("Output file: %s" %args.output)