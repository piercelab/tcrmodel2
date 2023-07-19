import subprocess

from anarci import anarci


def get_seq(pdb_chain):
    command="grep -A1 %s data/databases/pdb_seqres.txt" % (pdb_chain)
    output = subprocess.getoutput(command)
    return output.split("\n")[1]


def parse_anarci(in_seq):
    try:
        command="ANARCI --scheme aho -i %s" % in_seq
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        output = output.decode("utf-8")  # decode bytes to string
    except subprocess.CalledProcessError as e:
        print(f"ANARCI failed for {in_seq} with error: {e.output.decode('utf-8')}")
        return "NA", "NA"

    cdr3, seq="",""
    for i in output.split("\n"):
        if i and i[0] != "#" and i[0] != "/":
            _, num, res = i.rstrip().split()
            num = int(num)
            if res != "-":
                seq += res
                if num >= 106 and num <= 139:
                    cdr3 += res
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


def parse_tcr(in_seq):
    cdr3, seq = parse_anarci(in_seq)
    v_gene, j_gene = get_germlines(in_seq)
    return [cdr3, seq, v_gene, j_gene]

