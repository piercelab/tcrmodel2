import subprocess


three_to_one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
one_to_three = {value : key for (key, value) in three_to_one.items()}

def extract_sequence(pdb_fn):
    # get sequence from pdb
    # return a list of sequences for each chain
    seqs=[]
    pdb = []
    with open(pdb_fn) as fh:
        for line in fh:
            if line[0:4] == 'ATOM':
                pdb.append(line.rstrip())

    chains = []        
    for line in pdb:
        if line[0:4] == 'ATOM':
            chain_name = line[21]
            if (chain_name not in chains):
                chains.append(chain_name)
    res_count = 0
    start_res = 1
    for chain in chains:
        line_count = 0
        prev_res = ""
        seq=""
        for line in pdb:
            if line[21] == chain:
                curr_res = line[22:27]
                if line_count == 0:
                    prev_res = curr_res
                    seq+=three_to_one[line[17:20]]
                if curr_res != prev_res:
                    res_count += 1
                    prev_res = curr_res
                    seq+=three_to_one[line[17:20]]
                line_count += 1
        seqs.append(seq)

    return seqs


def anarci(seq):
    command="ANARCI --scheme aho -i %s" % seq
    output = subprocess.getoutput(command)
    return output

def gen_res_str(res):
    res_len = len(str(res))
    res_str = ""
    for i in range(4-res_len):
        res_str += " "
    res_str += str(res) + " "
    return res_str

def rename_chains_start_A_and_1(pdb):
    #get unique chain name and the residue start

    chains = []        
    for line in pdb:
        if line[0:4] == 'ATOM':
            chain_name = line[21]
            if (chain_name not in chains):
                chains.append(chain_name)
    chain_dict={}
    for i, j in enumerate(chains):
        chain_dict[j]=chr(ord('@')+i+1)

    pdb_renum = []
    for chain in chains:
        line_count = 0
        res_count = 0
        prev_res = ""
        start_res = 1
        for line in pdb:
            if line[21] == chain:
                curr_res = line[22:27]
                if line_count == 0:
                    prev_res = curr_res
                if curr_res != prev_res:
                    res_count += 1
                out_res = res_count + start_res 
                outline = line[:21] +chain_dict[chain]+ gen_res_str(out_res) + line[27:]
                pdb_renum.append(outline)
                prev_res = curr_res
                line_count += 1
    return pdb_renum

def align_pdbs_by_pmhc(ref_path, pdb_path, output_path, mhc_class):
    import MDAnalysis as mda 
    from MDAnalysis.analysis import align
    pdb_u = mda.Universe(pdb_path)
    ref = mda.Universe(ref_path)
    if mhc_class==1:
        align.alignto(pdb_u, ref, select="chainID C or chainID A", tol_mass=1000)
    if mhc_class==2:
        align.alignto(pdb_u, ref, select="chainID C or chainID A or chainID B", tol_mass=1000)

    pdb_u.select_atoms("protein").write(output_path)


def align_pdbs(ref_path, pdb_path, output_path):
    import MDAnalysis as mda 
    from MDAnalysis.analysis import align
    pdb_u = mda.Universe(pdb_path)
    ref = mda.Universe(ref_path)
    align.alignto(pdb_u, ref, tol_mass=1000)

    pdb_u.select_atoms("protein").write(output_path)


