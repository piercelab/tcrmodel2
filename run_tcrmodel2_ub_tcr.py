# Load required packages
import os
import sys
import pandas as pd
import json
from absl import flags
from absl import app
from glob import glob
import subprocess 
from anarci import anarci

from scripts import seq_utils,pdb_utils,tcr_utils,parse_tcr_seq

# input
flags.DEFINE_string('output_dir', "experiments/", 'Path to output directory.')
flags.DEFINE_string('tcra_seq', None, 'TCR alpha sequence')
flags.DEFINE_string('tcrb_seq', None, 'TCR beta sequence')
flags.DEFINE_string('job_id', "test001", 'Job id')
flags.DEFINE_string('ignore_pdbs_string', None, "Currently not supported")
flags.DEFINE_string('max_template_date', "2100-01-01", "Max template date, "
                    "format yyyy-mm-dd. Default to 2100-01-01.")
flags.DEFINE_bool('relax_structures', False, "Run amber minimization "
                  "on the structures")
flags.DEFINE_string("tp_db", "data/databases" , 
                    "Customized TCR pMHC database path")
flags.DEFINE_string("ori_db", None,
                    "Path to the database with pdb_mmcif database")
flags.DEFINE_integer("cuda_device", 1, 
                    "Visible cuda device number")
FLAGS = flags.FLAGS

def main(_argv):
    output_dir=FLAGS.output_dir
    tcra_seq=FLAGS.tcra_seq
    tcrb_seq=FLAGS.tcrb_seq
    job_id=FLAGS.job_id
    max_template_date=FLAGS.max_template_date
    relax_structures=FLAGS.relax_structures
    tp_db=FLAGS.tp_db
    ori_db=FLAGS.ori_db
    cuda_device=FLAGS.cuda_device

    if len(max_template_date)==0:
        max_template_date="2100-01-01"

    models_to_relax="none"
    if relax_structures==True:
        models_to_relax="all"

    # create output directory
    out_dir=os.path.join(output_dir,job_id)

    # make output directory
    os.makedirs(out_dir, exist_ok=True)

    # trim tcr sequence to variable domain only
    anarci_tcra=anarci([('tcra', tcra_seq)], scheme="aho", output=False)
    anarci_tcrb=anarci([('tcrb', tcrb_seq)], scheme="aho", output=False)
    tcra_seq="".join([item[-1] for item in anarci_tcra[0][0][0][0] if item[-1] != '-'])
    tcrb_seq="".join([item[-1] for item in anarci_tcrb[0][0][0][0] if item[-1] != '-'])

    # create fasta files 
    fasta_fn=os.path.join(out_dir, "%s.fasta" % job_id)
    fasta=">TCRa\n%s\n" % tcra_seq
    fasta+=">TCRb\n%s\n" % tcrb_seq
    with open(fasta_fn,'w+') as fh:
        fh.write(fasta)

    # create status file and update it
    status_file=os.path.join(out_dir,"modeling_status.txt")

    ###################
    # build structure #
    ###################
    databases=(f"--uniref90_database_path={tp_db}/uniref90.tcrmhc.fasta " 
            f"--mgnify_database_path={tp_db}/mgnify.fasta "
            f"--template_mmcif_dir={ori_db}/pdb_mmcif/mmcif_files/ "
            f"--obsolete_pdbs_path={ori_db}/pdb_mmcif/obsolete.dat "
            f"--small_bfd_database_path={tp_db}/small_bfd.tcrmhc.fasta "
            f"--pdb_seqres_database_path={tp_db}/pdb_seqres.txt "
            f"--uniprot_database_path={tp_db}/uniprot.tcrmhc.fasta "
            f"--data_dir={ori_db}")
    cmd=(f"python run_alphafold_tcrmodel2.3.py --db_preset=reduced_dbs "
         f"--fasta_paths={out_dir}/{job_id}.fasta " 
         f"--model_preset=multimer --output_dir={out_dir} {databases} "
         f"--max_template_date={max_template_date} "
         f"--use_gpu_relax={relax_structures} --models_to_relax={models_to_relax} "
         "--use_precomputed_msas=True --num_multimer_predictions_per_model=1 " 
         f"--save_template_names --status_file={status_file}")
    subprocess.run(cmd, shell=True)

    # renumber chains to start with A if not relax_structures
    if not relax_structures:
        models_list = [i for i in glob(f"{out_dir}/{job_id}/*.pdb") if os.path.basename(i).startswith('ranked')]
        for pdb_fn in models_list:
            pdb=[]
            with open(pdb_fn) as fh:
                for line in fh:
                    if line[0:4] == 'ATOM':
                        pdb.append(line.rstrip())
            pdb_renum=pdb_utils.rename_chains_start_A_and_1(pdb)
            pdb_renum_fn = pdb_fn.replace('.pdb', '_renum.pdb')
            with open(pdb_renum_fn,'w+') as fh:
                fh.write("\n".join(pdb_renum))
            subprocess.run("mv %s %s" % (pdb_renum_fn, pdb_fn), shell=True)


    ####################
    # Parse statistics #
    ####################
    out_json={}
    
    #get scores
    items=['model_confidence','plddt','ptm','iptm']

    with open("%s/%s/model_scores.txt" % (out_dir, job_id)) as fh:
        for idx, line in enumerate(fh):
            scores=line.rstrip().split("\t")
            out_json["ranked_%d" % (idx)]={
                items[0]:scores[0],
                items[1]:scores[1],
                items[2]:scores[2],
                items[3]:scores[3]
            }

    #get templates
    def get_template(tmplt_path):
        tmplts=[]
        N=0
        with open(tmplt_path) as fh:
            for line in fh:
                if N==4:
                    break
                tmplts.append(line.rstrip())
                N+=1
        return tmplts

    tmplt_path_prefix="%s/%s/msas" % (out_dir, job_id)
    out_json["tcra_tmplts"]=get_template("%s/A/template_names.txt" % tmplt_path_prefix)
    out_json["tcrb_tmplts"]=get_template("%s/B/template_names.txt" % tmplt_path_prefix)


    # clean up unwanted files
    subprocess.run("mv %s/%s/ranked*pdb %s/; " % (out_dir, job_id, out_dir), shell=True)
    subprocess.run("rm -rf %s/%s*; " % (out_dir, job_id), shell=True)


    ####################
    # # Renumber output  #
    ####################
    
    models_list = [i for i in glob('%s/*' % (out_dir)) if os.path.basename(i).startswith('ranked')]
    for model in models_list:
        tcr_utils.renumber_tcr_pdb(model, '%s/%s' % (out_dir, os.path.basename(model)))

    # align all to ranked_0's pMHC
    try:
        models_list = [i for i in glob(f"{out_dir}/*.pdb") if os.path.basename(i).startswith('ranked')]
        ref="%s/ranked_0.pdb" % out_dir
        for pdb in models_list:
            pdb_aln = pdb.replace('.pdb', '_aln.pdb')
            pdb_utils.align_pdbs(ref, pdb, pdb_aln)
            subprocess.run("mv %s %s" % (pdb_aln, pdb), shell=True)
    except:
        print("unable to align pdbs")

    #parse tcr template sequences
    tcr_out_json={}
    for chain in "ab":
        tcr_key = f"tcr{chain}_seqs"
        tcr_out_json[tcr_key] = {}
        for pdb_chain in out_json[f"tcr{chain}_tmplts"]:
            in_seq = parse_tcr_seq.get_seq(pdb_chain)
            tcr_out_json[tcr_key][pdb_chain] = parse_tcr_seq.parse_tcr(in_seq)

    tcr_out_json["tcra_user"] = parse_tcr_seq.parse_tcr(tcra_seq)
    tcr_out_json["tcrb_user"] = parse_tcr_seq.parse_tcr(tcrb_seq)


    tcr_json_output_path = os.path.join(out_dir, 'tcr_seqs.json')
    with open(tcr_json_output_path, 'w') as f:
        f.write(json.dumps(tcr_out_json, indent=4))


    #get CDR3s confidence scores 
    def get_cdr3_conf(pdb_path):
       import MDAnalysis as mda
       pdb_u = mda.Universe(pdb_path)
       cdr3a_bfactors_avg = pdb_u.select_atoms('chainID D and resid 106:139').atoms.bfactors.mean()
       cdr3b_bfactors_avg = pdb_u.select_atoms('chainID E and resid 106:139').atoms.bfactors.mean()
       return cdr3a_bfactors_avg, cdr3b_bfactors_avg

    models_list = [i for i in glob('%s/*' % (out_dir)) if os.path.basename(i).startswith('ranked')]
    for model in models_list:
       cdr3a_bfactors_avg, cdr3b_bfactors_avg = get_cdr3_conf(model)
       out_json[os.path.basename(model).split('.pdb')[0]]['cdr3a_plddt'] = cdr3a_bfactors_avg
       out_json[os.path.basename(model).split('.pdb')[0]]['cdr3b_plddt'] = cdr3b_bfactors_avg

    json_output_path = os.path.join(out_dir, 'statistics.json')
    with open(json_output_path, 'w') as f:
        f.write(json.dumps(out_json, indent=4))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass