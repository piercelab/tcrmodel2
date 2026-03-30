import json
import os
import subprocess
import sys
from glob import glob

from absl import app, flags

from scripts import pdb_utils, pmhc_templates, seq_utils, tcr_utils


flags.DEFINE_string('output_dir', "experiments/",
                    'Path to output directory.')
flags.DEFINE_string('pep_seq', None, 'Peptide sequence')
flags.DEFINE_string(
    'mhca_seq',
    None,
    'MHC alpha sequence. If your target is a class I TCR-pMHC complex, then this input should contain the alpha 1 and '
    'alpha 2 domain sequence. If your target is a class II TCR-pMHC complex, then this input should contain alpha 1 '
    'domain sequence. If your input has more than the above-mentioned domain(s), the function seq_utils.trim_mhc will '
    'trim the input sequence down to the desired domains.',
)
flags.DEFINE_string(
    'mhcb_seq',
    None,
    'MHC beta sequence. Leave this argument blank, or leave it out completely if your target is a class I TCR-pMHC '
    'complex. If your target is a class II TCR-pMHC complex, this input should contain beta 1 domain sequence. If your '
    'input has more than the above-mentioned domain(s), the function seq_utils.trim_mhc will trim the input sequence '
    'down to the desired domains.',
)
flags.DEFINE_string('job_id', "test001", 'Job id')
flags.DEFINE_string(
    'ignore_pdbs_string',
    None,
    "Do not use these pdbs as pmhc templates, comma seperated pdb string, no space in between. Can be upper or lower "
    "case. ",
)
flags.DEFINE_string('max_template_date', "2100-01-01", "Max template date, format yyyy-mm-dd. Default to 2100-01-01.")
flags.DEFINE_bool('relax_structures', False, "Run amber minimization on the structures")
flags.DEFINE_string("tp_db", "data/databases", "Customized TCR pMHC database path")
flags.DEFINE_string("ori_db", None, "Path to AlphaFold database with pdb_mmcif and params")
flags.DEFINE_integer("cuda_device", 1, "Visible cuda device number")

FLAGS = flags.FLAGS


def main(_argv):
    output_dir = FLAGS.output_dir
    pep_seq = FLAGS.pep_seq
    mhca_seq = FLAGS.mhca_seq
    mhcb_seq = FLAGS.mhcb_seq
    job_id = FLAGS.job_id
    ignore_pdbs_string = FLAGS.ignore_pdbs_string
    max_template_date = FLAGS.max_template_date
    relax_structures = FLAGS.relax_structures
    tp_db = FLAGS.tp_db
    ori_db = FLAGS.ori_db

    if len(max_template_date) == 0:
        max_template_date = "2100-01-01"

    models_to_relax = "none"
    if relax_structures:
        models_to_relax = "all"

    # process ignore_pdb list
    ignore_pdbs = []
    if ignore_pdbs_string:
        try:
            ignore_pdbs = [pdb.lower() for pdb in ignore_pdbs_string.split(",")]

        except:
            ignore_pdbs = []

    # create output directory
    out_dir = os.path.join(output_dir, job_id)
    os.makedirs(out_dir, exist_ok=True)

    # check MHC class of the complex
    mhc_cls = 1
    if mhcb_seq:
        mhc_cls = 2

    # check peptide length of the user input
    pep_len = len(pep_seq)
    if mhc_cls == 1:
        if pep_len < 8 or pep_len > 15:
            print(
                f"It looks like your input peptide is {pep_len} amino acids long. For class I TCR-pMHC complexes, "
                "kindly ensure the peptide length is between 8-15."
            )
            sys.exit()

    else:
        if pep_len != 11:
            print(
                f"It looks like your input peptide is {pep_len} amino acids (aa) long. For class II TCR-pMHC "
                "complexes, kindly ensure that the peptide input is 11 aa in length. Specifically, it should consist "
                "of a 9 aa core with an additional 1 aa at both the N-terminal and C-terminal of the core peptide."
            )
            sys.exit()

    # trim mhc sequence to relevant domains only
    if mhc_cls == 1:
        try:
            mhca_seq = seq_utils.trim_mhc(mhca_seq, "1", ".", out_dir)

        except:
            print("Fail to identify alpha 1 and alpha 2 domain sequence in the 'mhca_seq' "
                  "input of your class I MHC target.")
            sys.exit()

    else:
        try:
            mhca_seq = seq_utils.trim_mhc(mhca_seq, "2", ".", out_dir)

        except:
            print("Fail to identify alpha 1 domain sequence in the 'mhca_seq' "
                  "input of your class II MHC target. If your input target is a class I "
                  "TCR-pMHC complex, then mhcb_seq variable should be left empty or left "
                  "out completely.")
            sys.exit()

        try:
            mhcb_seq = seq_utils.trim_mhc(mhcb_seq, "3", ".", out_dir)

        except:
            print(
                "Fail to identify beta 1 domain sequence in the 'mhcb_seq' "
                "input of your class II MHC target. If your input target is a class I "
                "TCR-pMHC complex, then mhcb_seq variable should be left empty or left "
                "out completely."
            )
            sys.exit()

    # build pmhc templates
    if mhc_cls == 1:
        pmhc_templates.gen_align_file_cls1(pep_seq, mhca_seq, out_dir, ignore_pdbs, max_template_date)

    else:
        pmhc_templates.gen_align_file_corep1_cls2(pep_seq, mhca_seq, mhcb_seq, out_dir, ignore_pdbs, max_template_date)

    # create fasta files
    fasta_fn = os.path.join(out_dir, "%s.fasta" % job_id)
    pmhc_oc_fasta_fn = os.path.join(out_dir, "%s_pmhc_oc.fasta" % job_id)

    fasta = ">Peptide\n%s\n" % pep_seq
    fasta += ">MHCa\n%s\n" % mhca_seq

    if mhc_cls == 2:
        fasta += ">MHCb\n%s\n" % mhcb_seq

    pmhc_oc_fasta = ">pMHC\n%s:%s" % (pep_seq, mhca_seq)
    if mhc_cls == 2:
        pmhc_oc_fasta += ":%s\n" % mhcb_seq

    with open(fasta_fn, 'w+') as fh:
        fh.write(fasta)

    with open(pmhc_oc_fasta_fn, 'w+') as fh:
        fh.write(pmhc_oc_fasta)

    # create status file and update it
    status_file = os.path.join(out_dir, "modeling_status.txt")

    ###############
    # build MSA #
    ###############
    with open(status_file, 'a') as fh:
        fh.write('Building MSAs...\n')

    template_string = ",,,"
    if mhc_cls == 2:
        template_string = ",,,,"

    databases = (
        f"--uniref90_database_path={tp_db}/uniref90.tcrmhc.fasta "
        f"--mgnify_database_path={tp_db}/mgnify.fasta "
        f"--template_mmcif_dir={ori_db}/pdb_mmcif/mmcif_files/ "
        f"--obsolete_pdbs_path={ori_db}/pdb_mmcif/obsolete.dat "
        f"--small_bfd_database_path={tp_db}/small_bfd.tcrmhc.fasta "
        f"--pdb_seqres_database_path={tp_db}/pdb_seqres.txt "
        f"--uniprot_database_path={tp_db}/uniprot.tcrmhc.fasta "
        f"--data_dir={ori_db}"
    )
    cmd = (
        "python run_alphafold_tcrmodel2.3.py --db_preset=reduced_dbs "
        f"--fasta_paths={out_dir}/{job_id}.fasta "
        f"--model_preset=multimer --output_dir={out_dir} {databases} "
        f"--max_template_date={max_template_date} --use_gpu_relax=False "
        "--save_msa_features_only --gen_feats_only "
        "--models_to_relax=none --feature_prefix=msa "
        "--save_template_names --use_custom_templates "
        f"--template_alignfile={template_string}"
    )
    subprocess.run(cmd, shell=True)

    # remove unwanted files to save space
    subprocess.run("rm -rf %s/%s/msas/" % (out_dir, job_id), shell=True)

    with open(status_file, 'a') as fh:
        fh.write('Building Structures...\n')

    ###################
    # build structure #
    ###################
    cmd = (
        f"python run_alphafold_tcrmodel2.3.py --db_preset=reduced_dbs "
        f"--fasta_paths={out_dir}/{job_id}_pmhc_oc.fasta "
        f"--model_preset=multimer --output_dir={out_dir} {databases} "
        f"--use_custom_templates --template_alignfile={out_dir}/pmhc_alignment.tsv "
        f"--max_template_date={max_template_date} "
        f"--use_gpu_relax={relax_structures} "
        f"--models_to_relax={models_to_relax} --use_precomputed_msas=True "
        "--num_multimer_predictions_per_model=1  --save_template_names "
        "--has_gap_chn_brk --msa_mode=single_sequence --iptm_interface=1:1:2 "
        f"--substitute_msa={out_dir}/{job_id}/msa_features.pkl "
        f"--status_file={status_file}"
    )
    subprocess.run(cmd, shell=True)

    # renumber chains to start with A if not relax_structures
    if not relax_structures:
        models_list = [i for i in glob(f"{out_dir}/{job_id}_pmhc_oc/*.pdb") if os.path.basename(i).startswith('ranked')]
        for pdb_fn in models_list:
            pdb = []

            with open(pdb_fn) as fh:
                for line in fh:
                    if line[0:4] == 'ATOM':
                        pdb.append(line.rstrip())

            pdb_renum = pdb_utils.rename_chains_start_A_and_1(pdb)
            pdb_renum_fn = pdb_fn.replace('.pdb', '_renum.pdb')

            with open(pdb_renum_fn, 'w+') as fh:
                fh.write("\n".join(pdb_renum))

            subprocess.run("mv %s %s" % (pdb_renum_fn, pdb_fn), shell=True)

    ####################
    # Parse statistics #
    ####################
    out_json = {}

    # Get scores
    items = ['ranking_confidence', 'plddt', 'ptm', 'iptm']

    with open("%s/%s_pmhc_oc/model_scores.txt" % (out_dir, job_id)) as fh:
        for idx, line in enumerate(fh):
            scores = line.rstrip().split("\t")
            out_json["ranked_%d" % (idx)] = {
                items[0]: scores[0],
                items[1]: scores[1],
                items[2]: scores[2],
                items[3]: scores[3],
            }

    def get_template(tmplt_path):
        tmplts = []
        N = 0
        with open(tmplt_path) as fh:
            for line in fh:
                if N == 4:
                    break

                tmplts.append(line.rstrip())
                N += 1

        return tmplts

    tmplt_path_prefix = "%s/%s_pmhc_oc/msas" % (out_dir, job_id)
    out_json["pmhc_tmplts"] = get_template("%s/A/template_names.txt" % tmplt_path_prefix)

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
        tcr_utils.renumber_mhc_pdb(model, '%s/%s' % (out_dir, os.path.basename(model)), mhc_cls)

    # align all to ranked_0's pMHC
    try:
        models_list = [i for i in glob(f"{out_dir}/*.pdb") if os.path.basename(i).startswith('ranked')]
        ref = "%s/ranked_0.pdb" % out_dir
        for pdb in models_list:
            pdb_aln = pdb.replace('.pdb', '_aln.pdb')
            pdb_utils.align_pdbs_by_pmhc(ref, pdb, pdb_aln, mhc_cls)

            subprocess.run("mv %s %s" % (pdb_aln, pdb), shell=True)

    except:
        print("unable to align pdbs")

    # Write statistics
    json_output_path = os.path.join(out_dir, 'statistics.json')
    with open(json_output_path, 'w') as f:
        f.write(json.dumps(out_json, indent=4))


if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass
