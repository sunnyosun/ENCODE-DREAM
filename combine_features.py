"""
This script combines features from different files

By Xiaoji Sun
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

# combine motif and rnaseq features
def motif_tss_dnase (motifscore_filename, tssrnaseq_filename, dnase_filename, label_filename, outname, cellline):
	motif=pd.read_csv(motifscore_filename,sep='\t',header=None,engine='python')
	motif.columns=['chr','start','end','motif_score']
	tss=pd.read_csv(tssrnaseq_filename,sep='\t',header=None,engine='python')
	tss.columns=['chr','start','end','nearest_tss_chr','nearest_tss_start','nearest_tss_end','gene','strand','logTPM','distance']
	dnase=pd.read_csv(dnase_filename,sep='\t',header=None,engine='python')
	label=pd.read_csv(label_filename,sep='\t',header=None,engine='python')
	label.columns=['chr','start','end','A549','HeLa-S3','K562','MCF-7']
	comb=pd.concat([motif.loc[:,'chr'],tss.loc[:,'start'],tss.loc[:,'end'],motif.loc[:,'motif_score'],tss.loc[:,'logTPM'],tss.loc[:,'distance']],axis=1)
	distance=[float(i)/1000 for i in comb.distance]
	comb.distance=distance
	comb_all=pd.concat([comb,dnase.loc[:,3:15],label.loc[:,cellline]],axis=1)
	comb_all.columns=['chr','start','end','motif_score','logTPM','distance','minus600_minus400','minus500_minus300','minus400_minus200','minus300_minus100','minus200_0','minus100_targetmid','target_region','targetmid_100','plus0_200','plus100_300','plus200_400','plus300_500','plus400_600',cellline]
	comb_all.to_csv('combined_features_'+outname+'.txt', sep='\t',index=False)


#########################################################################
# Function main
def main():
    """
    This function sets the whole script will execute if run at the commandline.
    """
    # assume the input filename is the first argument
    motifscore_filename = sys.argv[1]
    tssrnaseq_filename = sys.argv[2]
    dnase_filename = sys.argv[3]
    label_filename = sys.argv[4]
    outname=sys.argv[5]
    cellline=sys.argv[6]

    #outname=motifscore_filename.split('/')[-1].split('_')[-1]
    if len(sys.argv) == 7:
    	motif_tss_dnase (motifscore_filename, tssrnaseq_filename, dnase_filename, label_filename, outname, cellline)

# this will be executed when the script is run    
if __name__=='__main__':
    main()



