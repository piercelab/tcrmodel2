# Load required packages
import os
import sys
# import pandas as pd
import json
from absl import flags
from absl import app
from glob import glob
import subprocess

from scripts import pmhc_templates, seq_utils, tcr_utils, pdb_utils, parse_tcr_seq

# input
flags.DEFINE_string('output_dir', "experiments/", 
                    'Path to output directory.')
flags.DEFINE_string('tcra_seq', None, 'TCR alpha sequence')
flags.DEFINE_string('tcrb_seq', None, 'TCR beta sequence')
flags.DEFINE_string('pep_seq', None, 'Peptide sequence')
flags.DEFINE_string('mhca_seq', None, 'MHC alpha sequence')
flags.DEFINE_string('mhcb_seq', None, 'MHC beta sequence')
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
flags.DEFINE_integer("cuda_device", 1, 
                    "Visible cuda device number")
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
    cuda_device=FLAGS.cuda_device

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

    # # create output directory
    out_dir=os.path.join(output_dir,job_id)
    os.makedirs(out_dir, exist_ok=True)

    # # check MHC class of the complex
    mhc_cls=1
    if mhcb_seq:
        mhc_cls=2
    
    # trim tcr sequence to variable domain only
    tcra_seq=seq_utils.trim_tcr(tcra_seq)
    tcrb_seq=seq_utils.trim_tcr(tcrb_seq)
    # trim mhc sequence to relevant domains only
    if mhc_cls==1:
        mhca_seq=seq_utils.trim_mhc(mhca_seq, "1", ".", out_dir)
    else:
        mhca_seq=seq_utils.trim_mhc(mhca_seq, "2", ".", out_dir)
        mhcb_seq=seq_utils.trim_mhc(mhcb_seq, "3", ".", out_dir)

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
        for i in range(5):
            pdb_fn="%s/%s_pmhc_oc/ranked_%d.pdb" % (out_dir, job_id, i)
            pdb=[]
            with open(pdb_fn) as fh:
                for line in fh:
                    if line[0:4] == 'ATOM':
                        pdb.append(line.rstrip())
            pdb_renum=pdb_utils.rename_chains_start_A_and_1(pdb)
            pdb_renum_fn="%s/%s_pmhc_oc/ranked_%d_renum.pdb"% (out_dir, job_id, i)
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
                items[0]:scores[0],
                items[1]:scores[1],
                items[2]:scores[2],
                items[3]:scores[3],
                items[4]:scores[4]
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
        for i in range(1,5):
            pdb="%s/ranked_%d.pdb" % (out_dir,i)
            pdb_aln="%s/ranked_%d_aln.pdb" % (out_dir,i)

            ref="%s/ranked_0.pdb" % out_dir
            pdb_utils.align_pdbs_by_pmhc(ref, pdb, pdb_aln, mhc_cls)
            subprocess.run("mv %s %s" % (pdb_aln, pdb), shell=True)
    except:
        print("unable to align pdbs")

    # compute angle 

    def get_docking_angles(tcr_docking_angle_exec, target_pdb, mhc_type):
        result = subprocess.run([f'{tcr_docking_angle_exec} {target_pdb} {mhc_type}'], shell=True, capture_output=True, text=True)
        parse_result = result.stdout.split("\n")
        get_angles = [i for i in parse_result if i.startswith('ANGLES')][0].split("\t")
        docking_angle = float(get_angles[1])
        inc_angle = float(get_angles[2])
        return docking_angle, inc_angle

    dock_dict = {}
    inc_dict = {} 
    tcr_docking_angle_exec = '/piercehome/programs/tcr_docking_angle/tcr_docking_angle'     
    models_list = [i for i in glob('%s/*' % (out_dir)) if os.path.basename(i).startswith('ranked')]
    for model in models_list:
        if mhc_cls==1:
            dock_ang, inc_ang = get_docking_angles(tcr_docking_angle_exec=tcr_docking_angle_exec, target_pdb=model, mhc_type=0)
        else:
            dock_ang, inc_ang = get_docking_angles(tcr_docking_angle_exec=tcr_docking_angle_exec, target_pdb=model, mhc_type=1)
        dock_dict.update({os.path.basename(model):dock_ang})
        inc_dict.update({os.path.basename(model):inc_ang})        
    out_json['angles'] = {'docking_angle':dock_dict,'incident_angle':inc_dict}
    

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

    
    #write statistics
    json_output_path = os.path.join(out_dir, 'statistics.json')
    with open(json_output_path, 'w') as f:
        f.write(json.dumps(out_json, indent=4))

    #parse tcr template sequences 
    tcra=out_json["tcra_tmplts"]
    tcra_seqs={}
    for pdb_chain in tcra:
        in_seq=parse_tcr_seq.get_seq(pdb_chain)
        anarci_out=parse_tcr_seq.anarci_custom(in_seq)
        cdr3, seq=parse_tcr_seq.parse_anarci(anarci_out)
        v_gene, j_gene=parse_tcr_seq.get_germlines(in_seq)
        tcra_seqs[pdb_chain]=[cdr3, seq, v_gene, j_gene]

    tcrb=out_json['tcrb_tmplts']
    tcrb_seqs={}
    for pdb_chain in tcrb:
        in_seq=parse_tcr_seq.get_seq(pdb_chain)
        anarci_out=parse_tcr_seq.anarci_custom(in_seq)
        cdr3, seq=parse_tcr_seq.parse_anarci(anarci_out)
        v_gene, j_gene=parse_tcr_seq.get_germlines(in_seq)
        tcrb_seqs[pdb_chain]=[cdr3, seq, v_gene, j_gene]

    tcr_out_json={}
    tcr_out_json["tcra_seqs"]=tcra_seqs
    tcr_out_json["tcrb_seqs"]=tcrb_seqs

    anarci_out=parse_tcr_seq.anarci_custom(tcra_seq)
    cdr3, seq=parse_tcr_seq.parse_anarci(anarci_out)
    v_gene, j_gene=parse_tcr_seq.get_germlines(tcra_seq)
    tcr_out_json["tcra_user"]=[cdr3, seq, v_gene, j_gene]

    anarci_out=parse_tcr_seq.anarci_custom(tcrb_seq)
    cdr3, seq=parse_tcr_seq.parse_anarci(anarci_out)
    v_gene, j_gene=parse_tcr_seq.get_germlines(tcrb_seq)
    tcr_out_json["tcrb_user"]=[cdr3, seq, v_gene, j_gene]

    tcr_json_output_path = os.path.join(out_dir, 'tcr_seqs.json')
    with open(tcr_json_output_path, 'w') as f:
        f.write(json.dumps(tcr_out_json, indent=4))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
