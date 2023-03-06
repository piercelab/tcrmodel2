
import subprocess
import os 

def anarci(seq):
    command="ANARCI --scheme aho -i %s" % seq
    output = subprocess.getoutput(command)
    return output


def parse_anarci(output):
    #24-42 (CDR1), 57-76 (CDR2), and 107-138 (CDR3)
    seq=""
    for i in output.split("\n"):
        if i[0]!="#" and i[0]!="/":
            fields=i.rstrip().split()
            num=int(fields[1])
            res=fields[-1]
            if res!="-":
                seq+=res
    return seq

def trim_tcr(seq):
    # trim TCR sequence to variable domain sequence
    anarci_out=anarci(seq)
    return parse_anarci(anarci_out)

def trim_mhc(seq, type, root_dir, out_dir):
    # trim MHC sequence to alpha1 and alpha2 if class I (type 1)
    # trim MHC to alpha1 if class II alpha (type 2)
    # trim MHC to beta1 if class II beta (type 3)

    if type=="1":
        hmmfile = "%s/scripts/mhc_hmm/classI.hmm" % root_dir 
    elif type=="2":
        hmmfile = "%s/scripts/mhc_hmm/classII_alpha.hmm" % root_dir 
    elif type=="3":
        hmmfile = "%s/scripts/mhc_hmm/classII_beta.hmm" % root_dir 
    else:
        return "error"

    tmpfn="%s/tmp.fa" % out_dir
    with open(tmpfn, 'w') as fh:
        fh.write(">tmp\n%s\n" % seq)
    
    hmmcmd="hmmsearch --noali %s %s" % (hmmfile,tmpfn)
    hmmout=subprocess.getoutput(hmmcmd)

    os.remove(tmpfn)
    start=0
    end=0
    hmmout=hmmout.split("\n")
    for i,line in enumerate(hmmout):
        if "Domain annotation for each sequence" in line:
            line=hmmout[i+4]
            start=int(line[59:66])
            end=int(line[67:74])
            break
    if start == 0 or end == 0:
        return "none"
    else:
        start-=1
        return seq[start:end]
