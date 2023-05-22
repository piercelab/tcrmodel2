import json
import subprocess
import anarci
import sys
from anarci import anarci


def get_seq(pdb_chain):
    command="grep -A1 %s data/databases/pdb_seqres.txt" % (pdb_chain)
    output = subprocess.getoutput(command)
    return output.split("\n")[1]

def anarci_custom(seq):
    command="ANARCI --scheme aho -i %s" % seq
    output = subprocess.getoutput(command)
    return output

def parse_anarci(output):
    cdr3, seq="",""
    for i in output.split("\n"):
        if i[0]!="#" and i[0]!="/":
            fields=i.rstrip().split()
            num=int(fields[1])
            res=fields[-1]
            if res!="-":
                seq+=res
            if res!="-" and num >= 106 and num <= 139:
                cdr3+=res
    return cdr3, seq


def get_germlines(seq:str):
    '''
    Get the VJ germlines from TCRa or TCRb sequences
    '''
    input_seq = [('seq',seq)]
    try:
        results = anarci(input_seq, scheme="aho", output=False, assign_germline=True)
        v_gene = results[1][0][0]['germlines']['v_gene'][0][1]
        j_gene = results[1][0][0]['germlines']['j_gene'][0][1]
    except:
        v_gene = 'NA'
        j_gene = 'NA'
    return v_gene, j_gene
