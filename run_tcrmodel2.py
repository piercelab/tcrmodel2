# Load required packages
# import pandas as pd
import json
import os
import subprocess
import sys
from glob import glob

from absl import app, flags
from anarci import anarci

from scripts import parse_tcr_seq, pdb_utils, pmhc_templates, seq_utils, tcr_utils

# import shutil



# input
flags.DEFINE_string('output_dir', "experiments/", 
                    'Path to output directory.')
flags.DEFINE_string('tcra_seq', None, 'TCR alpha sequence')
flags.DEFINE_string('tcrb_seq', None, 'TCR beta sequence')
flags.DEFINE_string('pep_seq', None, 'Peptide sequence')
flags.DEFINE_string('mhca_seq', None, 'MHC alpha sequence. If your target is a class I '
                    'TCR-pMHC complex, then this input should contain the alpha 1 and '
                     'alpha 2 domain sequence. If your target is a class II TCR-pMHC '
                     'complex, then this input should contain alpha 1 domain sequence. '
                     'If your input has more than the above-mentioned domain(s), the function '
                     'seq_utils.trim_mhc will trim the input sequence down to the desired domains.')
flags.DEFINE_string('mhcb_seq', None, 'MHC beta sequence. Leave this argument blank, or '
                    'leave it out completely if your target is a class I TCR-pMHC complex. '
                    'If your target is a class II TCR-pMHC complex, this input should '
                    'contain beta 1 domain sequence. If your input has more than the '
                    'above-mentioned domain(s), the function seq_utils.trim_mhc will '
                    'trim the input sequence down to the desired domains.')
flags.DEFINE_string('job_id', "test001", 'Job id')
flags.DEFINE_string('ignore_pdbs_string', None, "Do not use these pdbs as pmhc "
                    "templates, comma seperated pdb string, no space in between. "
                    "Can be upper or lower case. ")
flags.DEFINE_string('max_template_date', "2100-01-01", "Max template date, "
                    "format yyyy-mm-dd. Default to 2100-01-01.")
flags.DEFINE_bool('relax_structures', False, "Run amber minimization "
                  "on the structures")
flags.DEFINE_string("tp_db", "data/databases" , 
                    "Customized TCR pMHC database path")
flags.DEFINE_string("ori_db", None,
                    "Path to AlphaFold database with pdb_mmcif and params")

FLAGS = flags.FLAGS

def main(_argv):
    output_dir=FLAGS.output_dir
    tcra_seq=FLAGS.tcra_seq
    tcrb_seq=FLAGS.tcrb_seq
    pep_seq=FLAGS.pep_seq
    mhca_seq=FLAGS.mhca_seq
    mhcb_seq=FLAGS.mhcb_seq
    job_id=FLAGS.job_id
    ignore_pdbs_string=FLAGS.ignore_pdbs_string
    max_template_date=FLAGS.max_template_date
    relax_structures=FLAGS.relax_structures
    tp_db=FLAGS.tp_db
    ori_db=FLAGS.ori_db

    if len(max_template_date)==0:
        max_template_date="2100-01-01"
        
    models_to_relax="none"
    if relax_structures==True:
        models_to_relax="all"
    # process ignore_pdb list
    ignore_pdbs=[]
    if ignore_pdbs_string:
        try:
            ignore_pdbs=[pdb.lower() for pdb in ignore_pdbs_string.split(",")]
        except:
            ignore_pdbs=[]

    # create output directory
    out_dir=os.path.join(output_dir,job_id)
    os.makedirs(out_dir, exist_ok=True)

    # check MHC class of the complex
    mhc_cls=1
    if mhcb_seq:
        mhc_cls=2

    # check peptide length of the user input
    pep_len = len(pep_seq)
    if mhc_cls==1:
        if pep_len < 8 or pep_len > 15:
            print(f"It looks like your input peptide is {pep_len} amino acids long. For class I TCR-pMHC complexes, kindly ensure the peptide length is between 8-15.")
            sys.exit()
    else:
        if pep_len != 11:
            print(f"It looks like your input peptide is {pep_len} amino acids (aa) long. For class II TCR-pMHC complexes, kindly ensure that the peptide input is 11 aa in length. Specifically, it should consist of a 9 aa core with an additional 1 aa at both the N-terminal and C-terminal of the core peptide.")
            sys.exit()
    
    # trim tcr sequence to variable domain only
    anarci_tcra=anarci([('tcra', tcra_seq)], scheme="aho", output=False)
    anarci_tcrb=anarci([('tcrb', tcrb_seq)], scheme="aho", output=False)
    tcra_seq="".join([item[-1] for item in anarci_tcra[0][0][0][0] if item[-1] != '-'])
    tcrb_seq="".join([item[-1] for item in anarci_tcrb[0][0][0][0] if item[-1] != '-'])

    # trim mhc sequence to relevant domains only
    if mhc_cls==1:
        try:
            mhca_seq=seq_utils.trim_mhc(mhca_seq, "1", ".", out_dir)
        except:
            print("Fail to identify alpha 1 and alpha 2 domain sequence in the 'mhca_seq' "
                  "input of your class I MHC target.")
            sys.exit()
    else:
        try:
            mhca_seq=seq_utils.trim_mhc(mhca_seq, "2", ".", out_dir)
        except:
            print("Fail to identify alpha 1 domain sequence in the 'mhca_seq' "
                  "input of your class II MHC target. If your input target is a class I "
                  "TCR-pMHC complex, then mhcb_seq variable should be left empty or left "
                  "out completely.")
            sys.exit()
        try:
            mhcb_seq=seq_utils.trim_mhc(mhcb_seq, "3", ".", out_dir)
        except:
            print("Fail to identify beta 1 domain sequence in the 'mhcb_seq' "
                  "input of your class II MHC target. If your input target is a class I "
                  "TCR-pMHC complex, then mhcb_seq variable should be left empty or left "
                  "out completely.")
            sys.exit()

    # build pmhc templates
    if mhc_cls==1:
        pmhc_templates.gen_align_file_cls1(pep_seq, mhca_seq, out_dir, ignore_pdbs, max_template_date)
    else:
        pmhc_templates.gen_align_file_corep1_cls2(pep_seq, mhca_seq, mhcb_seq, out_dir, ignore_pdbs, max_template_date)

    # create status file and update it
    status_file=os.path.join(out_dir,"modeling_status.txt")
    with open(status_file, 'w') as fh:
        fh.write("Template features for pMHC created! Now generating MSA features and TCR template features...\n")

    # create fasta files 
    fasta_fn=os.path.join(out_dir, "%s.fasta" % job_id)
    pmhc_oc_fasta_fn=os.path.join(out_dir, "%s_pmhc_oc.fasta" % job_id)

    fasta=">TCRa\n%s\n" % tcra_seq
    fasta+=">TCRb\n%s\n" % tcrb_seq
    fasta+=">Peptide\n%s\n" % pep_seq
    fasta+=">MHCa\n%s\n" % mhca_seq
    if mhc_cls==2:
        fasta+=">MHCb\n%s\n" % mhcb_seq

    pmhc_oc_fasta=">TCRa\n%s\n" % tcra_seq
    pmhc_oc_fasta+=">TCRb\n%s\n" % tcrb_seq
    pmhc_oc_fasta+=">pMHC\n%s:%s" % (pep_seq, mhca_seq)
    if mhc_cls==2:
        pmhc_oc_fasta+=":%s\n" % mhcb_seq

    with open(fasta_fn,'w+') as fh:
        fh.write(fasta)
    with open(pmhc_oc_fasta_fn,'w+') as fh:
        fh.write(pmhc_oc_fasta)

    ###############
    # build MSA #
    ###############

    template_string=",,,"
    if mhc_cls==2:
        template_string=",,,,"
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
         f"--max_template_date={max_template_date} --use_gpu_relax=False "
         f"--save_msa_features_only --gen_feats_only "
         f"--models_to_relax=none --feature_prefix=msa "
         f"--save_template_names --use_custom_templates "
         f"--template_alignfile={template_string}")
    subprocess.run(cmd, shell=True)

    # remove unwanted files to save space
    subprocess.run("rm -rf %s/%s/msas/" % (out_dir, job_id), shell=True)

    ###################
    # build structure #
    ###################
    # model_log_output=os.path.join(out_dir, "modeling_log.txt")
    cmd=(f"python run_alphafold_tcrmodel2.3.py --db_preset=reduced_dbs "
         f"--fasta_paths={out_dir}/{job_id}_pmhc_oc.fasta "
         f"--model_preset=multimer --output_dir={out_dir} {databases} "
         "--use_custom_templates --template_alignfile=UseDefaultTemplate,"
         f"UseDefaultTemplate,{out_dir}/pmhc_alignment.tsv "
         f"--max_template_date={max_template_date} "
         f"--use_gpu_relax={relax_structures} "
         f"--models_to_relax={models_to_relax} --use_precomputed_msas=True "
         "--num_multimer_predictions_per_model=1  --save_template_names "
         "--has_gap_chn_brk --msa_mode=single_sequence --iptm_interface=1:1:2 "
         f"--substitute_msa={out_dir}/{job_id}/msa_features.pkl "
         f"--status_file={status_file}" )
    subprocess.run(cmd, shell=True)

    # renumber chains to start with A if not relax_structures
    if not relax_structures:
        models_list = [i for i in glob(f"{out_dir}/{job_id}_pmhc_oc/*.pdb") if os.path.basename(i).startswith('ranked')]
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
    items=['ranking_confidence','plddt','ptm','iptm','tcr-pmhc_iptm']

    with open("%s/%s_pmhc_oc/model_scores.txt" % (out_dir, job_id)) as fh:
        for idx, line in enumerate(fh):
            scores=line.rstrip().split("\t")
            out_json["ranked_%d" % (idx)]={
                items[0]:float(scores[0]),
                items[1]:float(scores[1]),
                items[2]:float(scores[2]),
                items[3]:float(scores[3]),
                items[4]:float(scores[4])
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

    tmplt_path_prefix="%s/%s_pmhc_oc/msas" % (out_dir, job_id)
    out_json["tcra_tmplts"]=get_template("%s/A/template_names.txt" % tmplt_path_prefix)
    out_json["tcrb_tmplts"]=get_template("%s/B/template_names.txt" % tmplt_path_prefix)
    out_json["pmhc_tmplts"]=get_template("%s/C/template_names.txt" % tmplt_path_prefix)

    json_output_path = os.path.join(out_dir, 'statistics.json')
    with open(json_output_path, 'w') as f:
        f.write(json.dumps(out_json, indent=4))

    # clean up unwanted files
    subprocess.run("mv %s/%s_pmhc_oc/ranked*pdb %s/; " % (out_dir, job_id, out_dir), shell=True)
    subprocess.run("rm -rf %s/%s*; " % (out_dir, job_id), shell=True)
    subprocess.run("rm %s/pmhc_alignment.tsv; " % (out_dir), shell=True)
    
        
    ####################
    # Renumber output  #
    ####################
    
    models_list = [i for i in glob('%s/*' % (out_dir)) if os.path.basename(i).startswith('ranked')]
    for model in models_list:
        tcr_utils.renumber_pdb(model, '%s/%s' % (out_dir, os.path.basename(model)), mhc_cls)   

    # align all to ranked_0's pMHC
    try:
        models_list = [i for i in glob(f"{out_dir}/*.pdb") if os.path.basename(i).startswith('ranked')]
        ref="%s/ranked_0.pdb" % out_dir
        for pdb in models_list:
            pdb_aln = pdb.replace('.pdb', '_aln.pdb')
            pdb_utils.align_pdbs_by_pmhc(ref, pdb, pdb_aln, mhc_cls)
            subprocess.run("mv %s %s" % (pdb_aln, pdb), shell=True)
    except:
        print("unable to align pdbs")


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

    ## Calculating iplddt score
     
    def calc_iplddt(pdb_file):
         
         chn1= "ABC"
         chn2= "DE"
         dis_cut=4
         lowest=-1.00
         
         try:
             with open(pdb_file, 'r') as file:
                 pdb_lines = file.readlines()
         except IOError:
             sys.exit(f"unable to open file: {pdb_file}")
         
         chn1int = {char: 1 for char in chn1}
         chn2int = {char: 1 for char in chn2}
         
         dis_cutoff = float(dis_cut) ** 2 
         
         chn1_int_plddt = {}
         chn2_int_plddt = {}
             
         for line in pdb_lines:
                 if not line.startswith("ATOM"):
                     continue
                 if line[12] == "H" or line[13] == "H":
                     continue
                 res_num = line[22:27].strip()
                 chn_id = line[21].strip()
                 res_id = line[17:20].strip()
                 atm_id = line[12:16].strip()
                 plddt1 = float(line[60:66].strip())
                 
                 if chn1int.get(chn_id) == 1:
                     x1 = float(line[30:38].strip())
                     y1 = float(line[38:46].strip())
                     z1 = float(line[46:54].strip())
                     
                     for line2 in pdb_lines:
                         if not line2.startswith("ATOM"):
                             continue
                         if line2[12] == "H" or line2[13] == "H":
                             continue
                         
                         chn2 = line2[21].strip()
                         res_num2 = line2[22:27].strip()
                         res_id2 = line2[17:20].strip()
                         atm_id = line2[12:16].strip()
                         plddt2 = float(line2[60:66].strip())
                         
                         if chn2int.get(chn2) == 1:
                             x2 = float(line2[30:38].strip())
                             y2 = float(line2[38:46].strip())
                             z2 = float(line2[46:54].strip())
                             dist = ((x1 - x2)**2) + ((y1 - y2)**2) + ((z1 - z2)**2)
                             if dist < dis_cutoff:
                                 chn1_int_plddt[f"{res_num}\t{chn_id}\t{res_id}"] = plddt1
                                 chn2_int_plddt[f"{res_num2}\t{chn2}\t{res_id2}"] = plddt2
                             
         weighted_sum = sum(chn1_int_plddt.values()) + sum(chn2_int_plddt.values())
         counts = len(chn1_int_plddt) + len(chn2_int_plddt)
         
         final_score = lowest if counts == 0 else weighted_sum / counts
         final_score = "{:.2f}".format(final_score)
         
         return final_score
    
    models_list = [i for i in glob('%s/*' % (out_dir)) if os.path.basename(i).startswith('ranked')]
    for model in models_list:
       iplddt_score = calc_iplddt(model)
       out_json[os.path.basename(model).split('.pdb')[0]]['IpLDDT'] = iplddt_score
    
    #write statistics
    json_output_path = os.path.join(out_dir, 'statistics.json')
    with open(json_output_path, 'w') as f:
        f.write(json.dumps(out_json, indent=4))

    tcr_out_json = {}
    #parse tcr template sequences
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



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
