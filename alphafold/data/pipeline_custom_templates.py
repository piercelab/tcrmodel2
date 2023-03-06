# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
import pandas as pd
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np
from alphafold.data.custom_templates import create_single_template_features, compile_template_features, mk_mock_template
import tree

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features

def make_mock_msa_features(msas) -> FeatureDict:
  """Constructs a empty feature dict of MSA features."""
  num_res = len(msas[0])
  deletion_matrices = [[0] * len(msas[0])]
  deletion_matrix = []
  int_msa = []
  species_ids = []
  seen_sequences = set()
  for sequence_index, sequence in enumerate(msas):
    # for sequence_index, sequence in enumerate(msa):
      # if sequence in seen_sequences:
      #   continue
      # seen_sequences.add(sequence)
    int_msa.append(
        [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
    deletion_matrix.append(deletion_matrices[sequence_index])
    species_ids.append("".encode('utf-8'))

  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features

def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  try:
    for msa_index, msa in enumerate(msas):
      if not msa:
        raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
      for sequence_index, sequence in enumerate(msa.sequences):
        if sequence in seen_sequences:
          continue
        seen_sequences.add(sequence)
        int_msa.append(
            [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
        deletion_matrix.append(msa.deletion_matrix[sequence_index])
        identifiers = msa_identifiers.get_identifiers(
            msa.descriptions[sequence_index])
        species_ids.append(identifiers.species_id.encode('utf-8'))
    num_res = len(msas[0].sequences[0])
  except:
    msa=msas
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))
    num_res = len(msas.sequences[0])

  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniref30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_binary_path=jackhmmer_binary_path
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniref30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas

  def custom_MSA_features(self,
                      MSA_database: str,
                      input_fasta_path: str,
                      msa_output_dir: str):
    custom_MSA_runner = jackhmmer.Jackhmmer(
      binary_path=self.jackhmmer_binary_path,
      database_path=MSA_database)
    custom_MSA_out_path = os.path.join(msa_output_dir, 'custom_MSA.sto')
    custom_MSA_result = run_msa_tool(
        msa_runner=custom_MSA_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=custom_MSA_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniref_max_hits)
    custom_msa = parsers.parse_stockholm(custom_MSA_result['sto'])
    logging.info('Custom MSA size: %d sequences.', len(custom_msa))

    return custom_msa
  
  def msa_for_default_template(self,
                                input_fasta_path: str,
                                msa_output_dir: str):
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_result = run_msa_tool(
        msa_runner=self.jackhmmer_uniref90_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=uniref90_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniref_max_hits)
    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])      
    logging.info('Uniref90 MSA size: %d sequences. This is for templates, '
                'not MSA construction)', len(uniref90_msa))
    return jackhmmer_uniref90_result,uniref90_msa

  def process(self, 
              input_fasta_path: str,
              msa_output_dir: str, 
              use_custom_templates: bool,
              template_alignfile: str, 
              msa_mode: str,
              use_custom_MSA_database: str,
              MSA_database: str,
              save_msa_fasta: bool,
              save_template_names: bool,
              msa_for_template_query_seq_only: bool) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    # print(input_fasta_str)
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    # print("length of input_seqs is %d:" % len(input_seqs[0]))
    # if len(input_seqs) != 1:
    #   raise ValueError(
    #       f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = "".join(input_seqs[0])
    logging.info("input_sequence is %s" % input_sequence)
    input_description = input_descs[0]
    num_res = len(input_sequence)

    ###########################################################
    ## MSA
    ###########################################################

    # if MSA used:
    if msa_mode!="single_sequence":
      if use_custom_MSA_database=="only":
        # custom MSA 
        custom_msa=DataPipeline.custom_MSA_features(self,MSA_database,input_fasta_path,msa_output_dir)
        msa_features = make_msa_features((custom_msa))
        if template_alignfile=="UseDefaultTemplate": 
          jackhmmer_uniref90_result,uniref90_msa=DataPipeline.msa_for_default_template(self,input_fasta_path,msa_output_dir)
      else:
        # default MSA
        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=self.jackhmmer_uniref90_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=uniref90_out_path,
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.uniref_max_hits)
        mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
        jackhmmer_mgnify_result = run_msa_tool(
            msa_runner=self.jackhmmer_mgnify_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=mgnify_out_path,
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.mgnify_max_hits)

        uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
        mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])


        if self._use_small_bfd:
          bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
          jackhmmer_small_bfd_result = run_msa_tool(
              msa_runner=self.jackhmmer_small_bfd_runner,
              input_fasta_path=input_fasta_path,
              msa_out_path=bfd_out_path,
              msa_format='sto',
              use_precomputed_msas=self.use_precomputed_msas)
          bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
        else:
          bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
          hhblits_bfd_uniclust_result = run_msa_tool(
              msa_runner=self.hhblits_bfd_uniref_runner,
              input_fasta_path=input_fasta_path,
              msa_out_path=bfd_out_path,
              msa_format='a3m',
              use_precomputed_msas=self.use_precomputed_msas)
          bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])

        if use_custom_MSA_database=="add":
          custom_msa=DataPipeline.custom_MSA_features(self,MSA_database,input_fasta_path,msa_output_dir)
          msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa, custom_msa))
        else:
          msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

        logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
        logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
        logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))

    # if single sequence yet use default template
    elif template_alignfile=="UseDefaultTemplate": 
      jackhmmer_uniref90_result,uniref90_msa=DataPipeline.msa_for_default_template(self,input_fasta_path,msa_output_dir)
      msas=[input_sequence]
      msa_features = make_mock_msa_features(msas)
      
    # if single sequence and not use default template 
    else:
      msas=[input_sequence]
      msa_features = make_mock_msa_features(msas)

    if save_msa_fasta:
      msa_outpath=os.path.join(msa_output_dir, 'msa_feat_gaptoU.fasta')
      with open(msa_outpath, 'w+') as fh:
        fh.write(">query"+"\n"+input_sequence+"\n")
        counter=1
        for seq in msa_features['msa']:
            seq=[residue_constants.ID_TO_HHBLITS_AA[num] for num in seq]
            # for x in range(len(seq)):
            counter+=1
            fh.write(">seq_"+str(counter)+"\n")
            out="".join(seq).replace("-","U")
            fh.write(out+"\n")
                
    ###########################################################
    ## Template
    ###########################################################
    # if defaullt template should be used 
    if (not use_custom_templates) or (use_custom_templates and template_alignfile=="UseDefaultTemplate"):
      msa_for_templates = jackhmmer_uniref90_result['sto']
      msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
      msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
          msa_for_templates)
      if msa_for_template_query_seq_only:
        '''BP 12/15/22 let's just keep the target (query) sequence'''
        msa_for_templates = parsers.truncate_stockholm_msa2(msa_for_templates, 1)

      if self.template_searcher.input_format == 'sto':
        pdb_templates_result = self.template_searcher.query(msa_for_templates)
      elif self.template_searcher.input_format == 'a3m':
        uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
        pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
      else:
        raise ValueError('Unrecognized template input format: '
                        f'{self.template_searcher.input_format}')

      pdb_hits_out_path = os.path.join(
          msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
      with open(pdb_hits_out_path, 'w') as f:
        f.write(pdb_templates_result)

      pdb_template_hits = self.template_searcher.get_template_hits(
          output_string=pdb_templates_result, input_sequence=input_sequence)
          
      templates_result = self.template_featurizer.get_templates(
      query_sequence=input_sequence,
      hits=pdb_template_hits)

    #if user defined template used
    else:
      #############################################
      #update the templates
      if len(template_alignfile)==0:
        templates_result=mk_mock_template(input_sequence)
        # print(templates_result.features['template_domain_names'])
      else:
        data = pd.read_table(template_alignfile)
        # cols = ('template_pdbfile target_to_template_alignstring identities '
        #         'target_len template_len'.split())
        template_features_list = []
        for tnum, row in data.iterrows():
            target_to_template_alignment = {
                int(x.split(':')[0]) : int(x.split(':')[1]) # 0-indexed
                for x in row.target_to_template_alignstring.split(';')
            }

            template_name = os.path.basename(row.template_pdbfile).split(".")[0]
            template_features = create_single_template_features(
                input_sequence, row.template_pdbfile, target_to_template_alignment,
                template_name, allow_chainbreaks=True, allow_skipped_lines=True,
                expected_identities = None,
                expected_template_len = row.template_len,
            )
            template_features_list.append(template_features)

        templates_result = compile_template_features(
            template_features_list)

      #############################################


    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    # if len(input_seqs[0]) > 1:
    if isinstance(input_seqs[0], list):
      # print("There are %d semi-colon seperated chains in this chain. Therefore updating the residue_index now" % len(input_seqs[0]))
      # Minkyung's code
      # add big enough number to residue index to indicate chain breaks
      idx_res = sequence_features['residue_index']
      Ls=[len(seq) for seq in input_seqs[0]]
      L_prev = 0
      # Ls: number of residues in each chain
      for L_i in Ls[:-1]:
          idx_res[L_prev + L_i:] += 200
          L_prev += L_i
      # chains = list("".join([ascii_uppercase[n] * L for n, L in enumerate(Ls)]))
      sequence_features['residue_index'] = idx_res

    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4, and mock/no '
                 'templates would also show number of templates as 1): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    if save_template_names:
      temp_name_fn=os.path.join(msa_output_dir, 'template_names.txt')
      template_names=[name.decode('utf-8') for name in templates_result.features['template_domain_names']]
      with open(temp_name_fn, 'w+') as fh:
          fh.write("\n".join(template_names))
          
    return {**sequence_features, **msa_features, **templates_result.features}
