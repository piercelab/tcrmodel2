import json
import subprocess
import anarci
import sys
from anarci import anarci

fn=sys.argv[1]
user_json=sys.argv[2]
json_output_path=sys.argv[3]

with open(fn) as fh:
    f=json.load(fh)

def get_seq(pdb_chain):
    command="grep -A1 %s /piercehome/alphafold/genetic_databases/pdb_seqres/pdb_seqres.txt" % (pdb_chain)
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

tcra=f['tcra_tmplts']
tcra_seqs={}
for pdb_chain in tcra:
    in_seq=get_seq(pdb_chain)
    anarci_out=anarci_custom(in_seq)
    cdr3, seq=parse_anarci(anarci_out)
    v_gene, j_gene=get_germlines(in_seq)
    tcra_seqs[pdb_chain]=[cdr3, seq, v_gene, j_gene]

tcrb=f['tcrb_tmplts']
tcrb_seqs={}
for pdb_chain in tcrb:
    in_seq=get_seq(pdb_chain)
    anarci_out=anarci_custom(in_seq)
    cdr3, seq=parse_anarci(anarci_out)
    v_gene, j_gene=get_germlines(in_seq)
    tcrb_seqs[pdb_chain]=[cdr3, seq, v_gene, j_gene]

out_json={}
out_json["tcra_seqs"]=tcra_seqs
out_json["tcrb_seqs"]=tcrb_seqs

with open(user_json) as fh:
    uf=json.load(fh)

aseq=uf['aseq_user']
anarci_out=anarci_custom(aseq)
cdr3, seq=parse_anarci(anarci_out)
v_gene, j_gene=get_germlines(in_seq)
out_json["tcra_user"]=[cdr3, seq, v_gene, j_gene]

bseq=uf['bseq_user']
anarci_out=anarci_custom(bseq)
cdr3, seq=parse_anarci(anarci_out)
v_gene, j_gene=get_germlines(in_seq)
out_json["tcrb_user"]=[cdr3, seq, v_gene, j_gene]

with open(json_output_path, 'w') as f:
    f.write(json.dumps(out_json, indent=4))
