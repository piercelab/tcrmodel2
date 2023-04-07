# Load required packages
import os
import sys
import pandas as pd
import json
from absl import flags
from absl import app
from glob import glob
import MDAnalysis as mda

fn=sys.argv[1]
out_dir=sys.argv[2]
json_output_path=sys.argv[3]

with open(fn) as fh:
    out_json=json.load(fh)

#get CDR3s confidence scores 
def get_cdr3_conf(pdb_path):
    pdb_u = mda.Universe(pdb_path)
    cdr3a_bfactors_avg = pdb_u.select_atoms('chainID D and resid 106:139').atoms.bfactors.mean()
    cdr3b_bfactors_avg = pdb_u.select_atoms('chainID E and resid 106:139').atoms.bfactors.mean()
    return cdr3a_bfactors_avg, cdr3b_bfactors_avg

models_list = [i for i in glob('%s/*' % (out_dir)) if os.path.basename(i).startswith('ranked')]
for model in models_list:
    cdr3a_bfactors_avg, cdr3b_bfactors_avg = get_cdr3_conf(model)
    out_json[os.path.basename(model).split('.pdb')[0]]['cdr3a_plddt'] = cdr3a_bfactors_avg
    out_json[os.path.basename(model).split('.pdb')[0]]['cdr3b_plddt'] = cdr3b_bfactors_avg

with open(json_output_path, 'w') as f:
    f.write(json.dumps(out_json, indent=4))
