#Line 1-190 is from https://github.com/phbradley/alphafold_finetune/blob/main/predict_utils.py
from alphafold.common import residue_constants
from alphafold.data import templates
import numpy as np

#this is from alphafold/data.templates.py
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, List
import dataclasses
@dataclasses.dataclass(frozen=True)
class TemplateSearchResult:
  features: Mapping[str, Any]
#   errors: Sequence[str]
#   warnings: Sequence[str]

def load_pdb_coords(
        pdbfile,
        allow_chainbreaks=False,
        allow_skipped_lines=False,
        verbose=False,
):
    ''' returns: chains, all_resids, all_coords, all_name1s
    '''

    chains = []
    all_resids = {}
    all_coords = {}
    all_name1s = {}

    if verbose:
        print('reading:', pdbfile)
    skipped_lines = False
    with open(pdbfile,'r') as data:
        for line in data:
            if (line[:6] in ['ATOM  ','HETATM'] and line[17:20] != 'HOH' and
                line[16] in ' A1'):
                if ( line[17:20] in residue_constants.restype_3to1
                     or line[17:20] == 'MSE'): # 2022-03-31 change to include MSE
                    name1 = ('M' if line[17:20] == 'MSE' else
                             residue_constants.restype_3to1[line[17:20]])
                    resid = line[22:27]
                    chain = line[21]
                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)
                    if line.startswith('HETATM'):
                        print('WARNING: HETATM', pdbfile, line[:-1])
                    atom = line[12:16].split()[0]
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = {}
                        all_name1s[chain][resid] = name1

                    all_coords[chain][resid][atom] = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                else:
                    print('skip ATOM line:', line[:-1], pdbfile)
                    skipped_lines = True

    # check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if 'C' in coords1 and 'N' in coords2:
                dis = np.sqrt(np.sum(np.square(coords1['C']-coords2['N'])))
                if dis>maxdis:
                    print('WARNING chainbreak:', chain, res1, res2, dis, pdbfile)
                    if not allow_chainbreaks:
                        print('STOP: chainbreaks', pdbfile)
                        print('DONE')
                        exit()

    if skipped_lines and not allow_skipped_lines:
        print('STOP: skipped lines:', pdbfile)
        print('DONE')
        exit()

    return chains, all_resids, all_coords, all_name1s

def fill_afold_coords(
        chain_order,
        all_resids,
        all_coords,
):
    ''' returns: all_positions, all_positions_mask

    these are 'atom37' coords (not 'atom14' coords)

    '''
    assert residue_constants.atom_type_num == 37 #HACK/SANITY
    crs = [(chain,resid) for chain in chain_order for resid in all_resids[chain]]
    num_res = len(crs)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                  dtype=np.int64)
    for res_index, (chain,resid) in enumerate(crs):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        for atom_name, xyz in all_coords[chain][resid].items():
            x,y,z = xyz
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name != 'NV': # PRO NV OK to skip
                # this is just debugging/verbose output:
                name = atom_name[:]
                while name[0] in '123':
                    name = name[1:]
                if name[0] != 'H':
                    print('unrecognized atom:', atom_name, chain, resid)
            # elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
            #     # Put the coordinates of the selenium atom in the sulphur column.
            #     pos[residue_constants.atom_order['SD']] = [x, y, z]
            #     mask[residue_constants.atom_order['SD']] = 1.0

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask
    return all_positions, all_positions_mask


def create_single_template_features(
        target_sequence,
        template_pdbfile,
        target_to_template_alignment,
        template_name, # goes into template_domain_names, .encode()'ed
        allow_chainbreaks=True,
        allow_skipped_lines=True,
        expected_identities=None,
        expected_template_len=None,
):
    num_res = len(target_sequence)
    chains_tmp, all_resids_tmp, all_coords_tmp, all_name1s_tmp = load_pdb_coords(
        template_pdbfile, allow_chainbreaks=allow_chainbreaks,
        allow_skipped_lines=allow_skipped_lines,
    )

    crs_tmp = [(c,r) for c in chains_tmp for r in all_resids_tmp[c]]
    num_res_tmp = len(crs_tmp)
    template_full_sequence = ''.join(all_name1s_tmp[c][r] for c,r in crs_tmp)

    # if expected_template_len:
    #     assert len(template_full_sequence) == expected_template_len

    all_positions_tmp, all_positions_mask_tmp = fill_afold_coords(
        chains_tmp, all_resids_tmp, all_coords_tmp)

    identities = sum(target_sequence[i] == template_full_sequence[j]
                     for i,j in target_to_template_alignment.items())
    if expected_identities:
        assert identities == expected_identities

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                  dtype=np.int64)

    template_alseq = ['-']*num_res
    for i,j in target_to_template_alignment.items(): # i=target, j=template
        template_alseq[i] = template_full_sequence[j]
        all_positions[i] = all_positions_tmp[j]
        all_positions_mask[i] = all_positions_mask_tmp[j]

    template_sequence = ''.join(template_alseq)
    assert len(template_sequence) == len(target_sequence)
    assert identities == sum(a==b for a,b in zip(template_sequence, target_sequence))

    template_aatype = residue_constants.sequence_to_onehot(
        template_sequence, residue_constants.HHBLITS_AA_TO_ID)

    template_features = {
        'template_all_atom_positions': all_positions,
        'template_all_atom_masks': all_positions_mask,
        'template_sequence': template_sequence.encode(),
        'template_aatype': template_aatype,
        'template_domain_names': template_name.encode(),
        'template_sum_probs': [identities],
        }
    return template_features


def compile_template_features(template_features_list):
    all_template_features = {}
    for name, dtype in templates.TEMPLATE_FEATURES.items():
        all_template_features[name] = np.stack(
            [f[name] for f in template_features_list], axis=0).astype(dtype)
    return TemplateSearchResult(features=all_template_features)


#This function is modified from https://github.com/sokrypton/ColabFold/blob/aa7284b56c7c6ce44e252787011a6fd8d2817f85/colabfold/batch.py
def mk_mock_template(query_sequence: Union[List[str], str], num_temp: int = 1):
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        # "template_confidence_scores": np.tile(
        #     output_confidence_scores[None], [num_temp, 1]
        # ),
        "template_domain_names": np.array([f"none".encode()] * num_temp),
        # "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    # all_template_features=compile_template_features([template_features])
    return TemplateSearchResult(features=template_features)


    # return all_template_features
