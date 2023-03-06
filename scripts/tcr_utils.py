from anarci import anarci
import numpy as np

def renumber_pdb(pdb_path, output_path, mhc_class=1):
        import MDAnalysis as mda 

        '''
        This function takes a pdb from tcrmodel output and renumber the residues according to the AHO numering scheme
        '''
        #load pdb 
        pdb_u = mda.Universe(pdb_path)
        #create list of chains per residue
        list_chains = [np.unique(i)[0] for i in pdb_u.residues.chainIDs]
        #create lsit of residues 
        list_res = [mda.lib.util.convert_aa_code(i) for i  in pdb_u.atoms.residues.resnames]
        #get tcra and tcrb sequence
        tcra_seq = ''.join(list(map(list_res.__getitem__, [i for i,a in enumerate(list_chains) if a == 'A']))) #considering that tcra is chain A in tcrmodel output
        tcrb_seq = ''.join(list(map(list_res.__getitem__, [i for i,a in enumerate(list_chains) if a == 'B']))) #considering that tcra is chain B in tcrmodel output
        #run anarci to get new residue numbers for tcra
        anarci_tcra = anarci([('tcra', tcra_seq)], scheme="aho", output=False)
        new_res_tcra = [i[0][0] for i in anarci_tcra[0][0][0][0] if i[1] != '-']
        #run anarci to get new residue numbers for tcrb
        anarci_tcrb = anarci([('tcrb', tcrb_seq)], scheme="aho", output=False)
        new_res_tcrb = [i[0][0] for i in anarci_tcrb[0][0][0][0] if i[1] != '-']
        #create copy
        tmp_numbering = pdb_u.residues.resids
        #change the residue number in original PDB for the new one
        tmp_numbering[np.array(list_chains) == 'A'] = new_res_tcra
        tmp_numbering[np.array(list_chains) == 'B'] = new_res_tcrb

        pdb_u.residues.resids = tmp_numbering

        chns=pdb_u.atoms.chainIDs
        chns[chns=="A"]="F"
        chns[chns=="B"]="G"
        chns[chns=="D"]="A"
        if mhc_class==2:
            chns[chns=="E"]="B"
        chns[chns=="F"]="D"
        chns[chns=="G"]="E"
        pdb_u.atoms.chainIDs=chns

        #write PDB
        pdb_u.select_atoms("protein").write(output_path)
    
def renumber_tcr_pdb(pdb_path, output_path):
        import MDAnalysis as mda 

        '''
        This function takes a pdb from tcrmodel output and renumber the residues according to the AHO numering scheme
        '''
        #load pdb 
        pdb_u = mda.Universe(pdb_path)
        #create list of chains per residue
        list_chains = [np.unique(i)[0] for i in pdb_u.residues.chainIDs]
        #create lsit of residues 
        list_res = [mda.lib.util.convert_aa_code(i) for i  in pdb_u.atoms.residues.resnames]
        #get tcra and tcrb sequence
        tcra_seq = ''.join(list(map(list_res.__getitem__, [i for i,a in enumerate(list_chains) if a == 'A']))) #considering that tcra is chain A in tcrmodel output
        tcrb_seq = ''.join(list(map(list_res.__getitem__, [i for i,a in enumerate(list_chains) if a == 'B']))) #considering that tcra is chain B in tcrmodel output
        #run anarci to get new residue numbers for tcra
        anarci_tcra = anarci([('tcra', tcra_seq)], scheme="aho", output=False)
        new_res_tcra = [i[0][0] for i in anarci_tcra[0][0][0][0] if i[1] != '-']
        #run anarci to get new residue numbers for tcrb
        anarci_tcrb = anarci([('tcrb', tcrb_seq)], scheme="aho", output=False)
        new_res_tcrb = [i[0][0] for i in anarci_tcrb[0][0][0][0] if i[1] != '-']
        #create copy
        tmp_numbering = pdb_u.residues.resids
        #change the residue number in original PDB for the new one
        tmp_numbering[np.array(list_chains) == 'A'] = new_res_tcra
        tmp_numbering[np.array(list_chains) == 'B'] = new_res_tcrb

        pdb_u.residues.resids = tmp_numbering

        chns=pdb_u.atoms.chainIDs
        chns[chns=="A"]="D"
        chns[chns=="B"]="E"

        pdb_u.atoms.chainIDs=chns

        #write PDB
        pdb_u.select_atoms("protein").write(output_path)
    
    