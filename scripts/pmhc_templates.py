import os
from scripts import pdb_utils
import pandas as pd
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import Align
from Bio.Align import substitution_matrices
from datetime import datetime


aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
tmplt_header="template_pdbfile\ttarget_to_template_alignstring\ttarget_len\ttemplate_len\tidentities\n"

def gen_aln(biopython_align_out,query_res_cnt,tmplt_res_cnt):
    # return a query-template alignment map
    alignment=""
    for index, ref_ab_res in enumerate(biopython_align_out[0]):
        if ref_ab_res=="-":  #query sequence residue not matched to template
            tmplt_res_cnt+=1
            continue
        md_res=biopython_align_out[1][index]
        if md_res=="-": # template residue not aligned to anything 
            query_res_cnt+=1
            continue
        else:
            alignment+="%d:%d;" % (query_res_cnt, tmplt_res_cnt)
            tmplt_res_cnt+=1
            query_res_cnt+=1
    return alignment[:-1]

def gen_pep_aln(pep_len):
    # return a query-template alignment map for peptide
    out=""
    for i in range(pep_len):
        out+="%d:%d;" % (i,i)
    return out

def get_best_hit_cls1(pep, mhc, df1, ignore_pdbs, cutoff):
    # get top ranking templates, ranked by MHC then peptide similarity
    # similarity is measured in BLOSUM62 alignment score 
    # for class I pMHCs, peptide of template must match query length
    df1['release_dates'] =  pd.to_datetime(df1['release_dates'], format='%Y-%m-%d') 

    tmplt_dict={} # keep track of alignment scores
    for idx, row in df1.iterrows():
        if row.release_dates > cutoff:
            # ignore pdbs released after a certain date
            continue
        if row.PDB.lower() in ignore_pdbs:
            # ignore pdbs blacklisted by user
            continue
        if len(row.pep)!=len(pep):
            # ignore templates with peptide of different length than query
            continue
        else:
            # obtain MHC alignment score
            #change penalize gaps back to 0:
            aligner.open_gap_score=0
            aligner.extend_gap_score=0
            alignments = aligner.align(mhc, row.MHC)
            alignment = alignments[0]
            mhc_score = alignment.score


            # obtain peptide alignment score

            #test penalize gaps, default is 0:
            aligner.open_gap_score=-100
            aligner.extend_gap_score=-100
            alignments = aligner.align(pep, row.pep)
            alignment = alignments[0]
            pep_score = alignment.score

            # update template dictionary with scores
            tmplt_dict[row.PDB]= (mhc_score,pep_score)

    # rank templates first by MHC then by peptide similarity 
    tmplt_dict=dict(sorted(tmplt_dict.items(), key=lambda item: item[1],reverse=True))

    # return top 4 templates 
    tmplt_dict=list(tmplt_dict.items())[:4]
    return tmplt_dict

def get_best_hit_cls2(pep, mhc1, mhc2, df2,ignore_pdbs, cutoff):
    # get top ranking templates, ranked by MHC then peptide similarity
    # similarity is measured in BLOSUM62 alignment score 
    # for class II MHCs, the MHC similarity is a sum of MHCa and MHCb similarity 
    df2['release_dates'] =  pd.to_datetime(df2['release_dates'], format='%Y-%m-%d') 

    tmplt_dict={} # keep track of alignment scores
    for idx, row in df2.iterrows():
        if row.release_dates > cutoff:
            # ignore pdbs released after a certain date
            continue
        if row.PDB.lower() in ignore_pdbs:
            # ignore pdbs blacklisted by user
            continue
        # obtain MHC alignment score
        mhc1_tmp, mhc2_tmp=row.MHC.split(" ")
        alignments = aligner.align(mhc1, mhc1_tmp)
        alignment = alignments[0]
        mhc1_score=alignment.score
        alignments = aligner.align(mhc2, mhc2_tmp)
        alignment = alignments[0]
        mhc2_score=alignment.score

        # obtain peptide alignment score
        pep_tmp=row.peptide_core
        alignments = aligner.align(pep_tmp, pep)
        alignment = alignments[0]
        pep_score=alignment.score

        # update template dictionary with scores
        tmplt_dict[row.PDB]=(mhc1_score+mhc2_score,pep_score)

    # rank templates first by MHC then by peptide similarity 
    tmplt_dict=dict(sorted(tmplt_dict.items(), key=lambda item: item[1],reverse=True))

    # return top 4 templates 
    tmplt_dict=list(tmplt_dict.items())[:4]
    return tmplt_dict

def gen_align_for_given_tmplt_cls1(tmplt_pdb, pep_len, mhc_seq):
    # generate alignment between query and template for a given template

    # extract template sequence
    tmplt_seq = pdb_utils.extract_sequence(tmplt_pdb)
    mhc_tmplt_seq=tmplt_seq[1]

    # query and template peptide aligned position-wise
    alignment=gen_pep_aln(pep_len)

    # align MHC by mapping template to query residues after pairwise alignment
    alignments=pairwise2.align.globalxx(mhc_seq, mhc_tmplt_seq, penalize_extend_when_opening=True)
    alignment+=gen_aln(alignments[0],pep_len,pep_len)

    tmplt_out="%s\t%s\t%d\t%d\t0\n" % (tmplt_pdb, alignment, pep_len+len(mhc_seq), len("".join(tmplt_seq)))
    return tmplt_out

def gen_align_for_given_tmplt_cls2(tmplt_pdb, pep_len, mhc1_seq, mhc2_seq):
    # generate alignment between query and template for a given template
    
    # extract template sequence
    tmplt_seq = pdb_utils.extract_sequence(tmplt_pdb)
    mhc1_tmplt_seq=tmplt_seq[1]
    mhc2_tmplt_seq=tmplt_seq[2]

    # align peptide position-wise
    alignment=gen_pep_aln(pep_len)

    # align MHC by mapping template to query residues after pairwise alignment
    alignments=pairwise2.align.globalxx(mhc1_seq, mhc1_tmplt_seq, penalize_extend_when_opening=True)
    alignment+=gen_aln(alignments[0],pep_len, pep_len)+";"
    alignments=pairwise2.align.globalxx(mhc2_seq, mhc2_tmplt_seq, penalize_extend_when_opening=True)
    alignment+=gen_aln(alignments[0],pep_len+len(mhc1_seq),pep_len+len(mhc1_tmplt_seq))

    tmplt_out="%s\t%s\t%d\t%d\t0\n" % (tmplt_pdb, alignment, pep_len+len(mhc1_seq)+len(mhc2_seq), len("".join(tmplt_seq)))
    return tmplt_out

def gen_align_file_cls1(
        pep_seq, 
        mhc_seq, 
        out_dir, 
        ignore_pdbs, 
        cutoff="2100-01-01"
        ):
    # generate full alignment file for a given query in class I

    # read all nr class I complexes as dataframe
    fn="/piercehome/tcr/TCRmodel-2.0/algorithm_2.3/data/templates/nr_cls_I_complexes.txt"
    df1=pd.read_csv(fn,sep="\t")
    cutoff = datetime.strptime(cutoff, '%Y-%m-%d')

    tmplt_dict=get_best_hit_cls1(pep_seq, mhc_seq, df1, ignore_pdbs, cutoff)
    tmplt_out=""
    for (tmplt_name,score) in tmplt_dict:
        tmplt_pdb="/piercehome/tcr/TCRmodel-2.0/algorithm_2.3/data/templates/pdb/%s.pmhc.pdb" % tmplt_name
        tmplt_out+=gen_align_for_given_tmplt_cls1(tmplt_pdb, len(pep_seq),mhc_seq)

    tmplt_aln_file=os.path.join(out_dir, "pmhc_alignment.tsv")    
    with open("%s" % tmplt_aln_file,'w+') as fh:
        fh.write("%s%s" % (tmplt_header, tmplt_out)) 

def gen_align_file_cls2(
        pep_seq, 
        mhc1_seq, 
        mhc2_seq, 
        out_dir, 
        ignore_pdbs,
        cutoff="2100-01-01"):
    # generate full alignment file for a given query in class II

    # read all nr class II complexes as dataframe
    fn="/piercehome/tcr/TCRmodel-2.0/algorithm_2.3/data/templates/nr_cls_II_complexes.txt"
    df2=pd.read_csv(fn,sep="\t")
    cutoff = datetime.strptime(cutoff, '%Y-%m-%d')

    tmplt_dict=get_best_hit_cls2(pep_seq, mhc1_seq, mhc2_seq, df2, ignore_pdbs, cutoff)
    tmplt_out=""
    for (tmplt_name,score) in tmplt_dict:
        tmplt_pdb="/piercehome/tcr/TCRmodel-2.0/algorithm_2.3/data/templates/pdb/%s.pmhc.pdb" % tmplt_name
        tmplt_out+=gen_align_for_given_tmplt_cls2(tmplt_pdb, len(pep_seq), mhc1_seq, mhc2_seq)

    tmplt_aln_file=os.path.join(out_dir, "pmhc_alignment.tsv")
    with open("%s" % tmplt_aln_file,'w+') as fh:
        fh.write("%s%s" % (tmplt_header, tmplt_out)) 
