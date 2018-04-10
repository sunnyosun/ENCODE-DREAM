"""
Find the cloest TSS

by Xiaoji Sun
"""

################################################################################
# Modules

# import regular expression module
import re

# import sys module
import sys

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import pandas as pd

##########################################################################
# Functions

# combine the RNA-seq data with TSS annotation
def rnaseq_tss(tss_filename, rnaseq_filename):
	# read in tss file
	f1=open(tss_filename,'r')
	tss=f1.readlines()
	f1.close()
	# match tss gene with rnaseq value
	out=[]
	for i in range(len(tss)):
		name=tss[i].replace(" ", "\t").split('\t')[3]
		f2=open(rnaseq_filename,'r')
		for j,line in enumerate(f2):
			if line.strip().split('\t')[0] == name:
				score=line.strip().split('\t')[1]
				out.append(tss[i].replace(" ", "\t").strip()+'\t'+score)
		f2.close()
	f=open(rnaseq_filename+'_tss','w+')
	f.write('\n'.join(str(a) for a in out))
	f.close()

# format the tss.sorted.bed file
def tss(tss_filename):
	# read in tss file
	f1=open(tss_filename,'r')
	tss=f1.readlines()
	f1.close()
	# match tss gene with rnaseq value
	out=[]
	for i in range(len(tss)):
		out.append(tss[i].replace(" ", "\t").strip())
	f=open('tss_sorted_new.bed','w+')
	f.write('\n'.join(str(a) for a in out))
	f.close()

# for the + and - strand files, find the closest gene
def closest_tss(plus_filename, minus_filename, outname):
	f1=open(plus_filename)
	plus=f1.readlines()
	f1.close()
	out=[]
	f2=open(minus_filename)
	for i, line in enumerate(f2):
		if int(plus[i].strip().split('\t')[4]) == -1:
			out.append('\t'.join(line.strip().split('\t')[:-1])+'\t'+str(abs(int(line.strip().split('\t')[9]))))
		elif int(line.strip().split('\t')[4]) == -1:
			out.append(plus[i].strip())
		else :
			score_plus=abs(int(plus[i].strip().split('\t')[9]))
			score_minus=abs(int(line.strip().split('\t')[9]))
			if score_plus > score_minus:
				out.append('\t'.join(line.strip().split('\t')[:-1])+'\t'+str(abs(int(line.strip().split('\t')[9]))))
			if score_plus <= score_minus:
				out.append(plus[i].strip())
	f2.close()
	f=open(outname+'.txt','w+')
	f.write('\n'.join(a for a in out))
	f.write('\n')
	f.close()


#########################################################################
# Function main
def main():
    """
    This function sets the whole script will execute if run at the commandline.
    """
    # assume the input filename is the first argument
    plus_filename = sys.argv[1]
    minus_filename = sys.argv[2]
    outname = sys.argv[3]
    if len(sys.argv) == 4:
    	closest_tss(plus_filename, minus_filename, outname)

# this will be executed when the script is run    
if __name__=='__main__':
    main()


