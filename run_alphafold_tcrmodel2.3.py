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

"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Union
root_dir="/piercehome/tcr/TCRmodel-2.0/algorithm_2.3"
sys.path.insert(1,root_dir)

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline_custom_templates
from alphafold.data import pipeline_multimer_custom_templates
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax

import jax.numpy as jnp
import numpy as np
import pickle as pkl

# Internal import (7716).

logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', "/scratch/Pierce/alphafold_v2.3_db/", 
                    'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path',None, 
                    'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None,
                    'Path to the MGnify database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path',None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', True, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_boolean('use_custom_templates', False, 'Whether to use custom '
                    'templates or not.')
flags.DEFINE_string('template_alignfile', None, 'The path to the custom template'
                    'files. If the target is a monomer, provide the template path '
                    'as-is. If a multimer, provide all template alignment files '
                    'the order they appear in the target, comma seperated. Leave '
                    'the path blank if no template should be used for a chain. '
                    'Write "UseDefaultTemplate" to use default alphafold pipeline '
                    'for generating the template for that chain.')
flags.DEFINE_string('msa_mode', None, 'Type "single_sequence" to not use any MSA')
flags.DEFINE_integer('num_recycle', 3, 'How many recycles')
flags.DEFINE_integer('num_ensemble', 1, 'How many ensembling iteractions')
flags.DEFINE_enum('use_custom_MSA_database', "none", ["none", "add", "only"], 'Whether to use custom '
                    'MSA database or not.')
flags.DEFINE_string('MSA_database', None, 'The path to the custom MSA database'
                    'files. If the target is a monomer, provide the template path '
                    'as-is. If a multimer, provide all template alignment files '
                    'the order they appear in the target, comma seperated.')
flags.DEFINE_string('run_model_names', None, 'Specify parameter name to run. This'
                    'is comma seperated alphafold parameter name. Only specified'
                    'model names will be run.')
flags.DEFINE_boolean('save_msa_fasta', False, 'Save msa features or not.')
flags.DEFINE_boolean('gen_feats_only', False, 'Only generate features and do not'
                    ' produce structure predictions.')
flags.DEFINE_boolean('save_template_names', False, 'Save template id to txt file.')
flags.DEFINE_boolean('has_gap_chn_brk', False, 'Have chain breaks introduced by ":".')
flags.DEFINE_string('substitute_msa', None, 'Path to feature.pkl whose MSA will '
                    'be used to substitute whatever MSA that will be generated by '
                    'this prediction round.')
flags.DEFINE_boolean('msa_for_template_query_seq_only', True, 'msa_for_template_query_seq_only')
flags.DEFINE_string('iptm_interface', None, 'iptm_interface')
flags.DEFINE_string('feature_prefix', None, 'Feature prefix')
flags.DEFINE_boolean('save_ranked_pdb_only', False, 'Do not save result pkl files '
                     'or unrelaxed pdbs, or relaxed pdbs that are not ranked.')
flags.DEFINE_boolean('save_msa_features_only', False, 'Only save msa related features.')
flags.DEFINE_string("status_file", None, 'Status report file path.')


FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output


def interface_parser(interfaces_string):
  interfaces=[]
  for interface in interfaces_string.split(","):
    interfaces.append([int(i) for i in interface.split(":")])
  return interfaces

def gen_res_str(res):
    res_len = len(str(res))
    res_str = ""
    for i in range(4-res_len):
        res_str += " "
    res_str += str(res) + " "
    return res_str

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline, pipeline_custom_templates.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: None,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax,
    use_custom_templates: bool,
    template_alignfile: str,
    msa_mode: str,
    use_custom_MSA_database: str,
    MSA_database: str,
    save_msa_fasta: bool,
    gen_feats_only: bool,
    save_template_names: bool,
    has_gap_chn_brk: bool,
    substitute_msa: str,
    msa_for_template_query_seq_only: bool,
    iptm_interface: str,
    feature_prefix: str,
    save_ranked_pdb_only: bool,
    save_msa_features_only: bool,
    status_file: str):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  if use_custom_templates or use_custom_MSA_database!="none":
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir,
        use_custom_templates=use_custom_templates,
        template_alignfile=template_alignfile,
        msa_mode=msa_mode,
        use_custom_MSA_database=use_custom_MSA_database,
        MSA_database=MSA_database,
        save_msa_fasta=save_msa_fasta,
        save_template_names=save_template_names,
        msa_for_template_query_seq_only=msa_for_template_query_seq_only)
  else:
    feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir,
      save_msa_fasta=save_msa_fasta,
      save_template_names=save_template_names,
      msa_for_template_query_seq_only=msa_for_template_query_seq_only)
  timings['features'] = time.time() - t_0

  # # Write out features as a pickled dictionary.
  if not save_ranked_pdb_only:
    features_output_path = os.path.join(output_dir, 'features.pkl')
    if feature_prefix:
      features_output_path = os.path.join(output_dir, '%s_features.pkl' % feature_prefix)
    if save_msa_features_only:
      key_save=['msa','deletion_matrix','cluster_bias_mask','bert_mask','msa_mask']
      feature_dict_new={}
      for key in key_save:
        feature_dict_new[key]=feature_dict[key]
      feature_dict=feature_dict_new
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)


  
  if save_msa_fasta:
    with open(os.path.join(output_dir, "all_msa_feat_gaptoU.fasta"), 'w+') as fh:
    # fh.write(">query"+"\n"+input_sequence+"\n")
      counter=1
      for seq in feature_dict['msa']:
          seq=[residue_constants.ID_TO_HHBLITS_AA[num] for num in seq]
          counter+=1
          fh.write(">seq_"+str(counter)+"\n")
          out="".join(seq).replace("-","U")
          fh.write(out+"\n")

  if gen_feats_only:
    return 
  if status_file:
    with open(status_file,"a") as fh:
      fh.write("All MSA and template features generated! Working on models now...\n")

  if substitute_msa:
    keys_substitute=['msa','deletion_matrix','cluster_bias_mask','bert_mask','msa_mask']
    with open(substitute_msa, 'rb') as fh:
      substitute_feature_dict = pkl.load(fh)
    for key in keys_substitute:
      feature_dict[key]=substitute_feature_dict[key]
      
  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  relaxed_pdbs = {}
  ranking_confidences = {}
  model_scores = {}
  relax_metrics = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    interfaces=[]
    if iptm_interface:
      interfaces=interface_parser(iptm_interface)

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed,
                                             interfaces=interfaces)

    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']
    model_scores[model_name] = [prediction_result['ranking_confidence'], np.mean(prediction_result['plddt'])]
    if 'iptm' in prediction_result:
      model_scores[model_name].append(prediction_result['ptm'])
      model_scores[model_name].append(prediction_result['iptm'])
    if "custom_iptm" in prediction_result:
      for score in prediction_result['custom_iptm']:
        model_scores[model_name].append(score)



    if not save_ranked_pdb_only:
      # Remove jax dependency from results.
      np_prediction_result = _jnp_to_np(dict(prediction_result))
      result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
      with open(result_output_path, 'wb') as f:
        pickle.dump(np_prediction_result, f, protocol=4)
    else:
      text_output_path = os.path.join(output_dir, 'model_%d_done' % model_index)
      with open(text_output_path,'w') as fh:
        fh.write("")
        
    if status_file:
      if model_index < 4:
        with open(status_file,"a") as fh:
          fh.write("Model %d generated! Currently working on model %d...\n" % (model_index+1, model_index+2))
      else:
        with open(status_file,"a") as fh:
          fh.write("Model %d generated! Wrapping things up now!\n" % (model_index+1))

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)


    if has_gap_chn_brk:
      #break chain
      prev_res=0
      new_res=0
      curr_chn=0
      chn_idx_adj=[]
      new_res_index=[]
      for res in unrelaxed_protein.residue_index:
          if res-prev_res>199:
              prev_res=res
              curr_chn+=1
              chn_idx_adj.append(curr_chn)
              new_res=1
          else:
              prev_res=res
              chn_idx_adj.append(curr_chn)
              new_res+=1
          new_res_index.append(new_res)
      chain_index=np.add(unrelaxed_protein.chain_index, np.array(chn_idx_adj))
      unrelaxed_protein= protein.Protein(
        aatype=unrelaxed_protein.aatype,
        atom_positions=unrelaxed_protein.atom_positions,
        atom_mask=unrelaxed_protein.atom_mask,
        residue_index=np.array(new_res_index,dtype=np.int32),
        chain_index=np.array(chain_index,dtype=np.int32),
        b_factors=unrelaxed_protein.b_factors) 

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    
    if not save_ranked_pdb_only:
      unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
      with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])



  # Rank by model confidence.
  ranked_order = [
      model_name for model_name, confidence in
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

  # Relax predictions.
  if models_to_relax == ModelsToRelax.BEST:
    to_relax = [ranked_order[0]]
  elif models_to_relax == ModelsToRelax.ALL:
    to_relax = ranked_order
  elif models_to_relax == ModelsToRelax.NONE:
    to_relax = []

  for model_name in to_relax:
    t_0 = time.time()
    relaxed_pdb_str, _, violations = amber_relaxer.process(
        prot=unrelaxed_proteins[model_name])
    relax_metrics[model_name] = {
        'remaining_violations': violations,
        'remaining_violations_count': sum(violations)
    }
    timings[f'relax_{model_name}'] = time.time() - t_0

    relaxed_pdbs[model_name] = relaxed_pdb_str

    if not save_ranked_pdb_only:
      relaxed_output_path = os.path.join(
          output_dir, f'relaxed_{model_name}.pdb')
      with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

  # Write out relaxed PDBs in rank order.
  for idx, model_name in enumerate(ranked_order):
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if model_name in relaxed_pdbs:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  model_scores_output=""
  for model in ranked_order:
    model_scores_output+="%s\n" % "\t".join(map(str,model_scores[model]))
  model_scores_output_path = os.path.join(output_dir, f'model_scores.txt')
  with open(model_scores_output_path, 'w') as f:
    f.write(model_scores_output)

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))
  if models_to_relax != ModelsToRelax.NONE:
    relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')
    
  if FLAGS.gen_feats_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  # _check_flag('small_bfd_database_path', 'db_preset',
  #             should_be_set=use_small_bfd)
  # _check_flag('bfd_database_path', 'db_preset',
  #             should_be_set=not use_small_bfd)
  # _check_flag('uniref30_database_path', 'db_preset',
  #             should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  # _check_flag('pdb70_database_path', 'model_preset',
  #             should_be_set=not run_multimer_system)
  # _check_flag('pdb_seqres_database_path', 'model_preset',
  #             should_be_set=run_multimer_system)
  # _check_flag('uniprot_database_path', 'model_preset',
  #             should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  if FLAGS.use_custom_templates or FLAGS.use_custom_MSA_database!="none":
    monomer_data_pipeline = pipeline_custom_templates.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas)

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    if FLAGS.use_custom_templates or FLAGS.use_custom_MSA_database!="none":
      data_pipeline = pipeline_multimer_custom_templates.DataPipeline(
          monomer_data_pipeline=monomer_data_pipeline,
          jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
          uniprot_database_path=FLAGS.uniprot_database_path,
          use_precomputed_msas=FLAGS.use_precomputed_msas)
    else:
      data_pipeline = pipeline_multimer.DataPipeline(
          monomer_data_pipeline=monomer_data_pipeline,
          jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
          uniprot_database_path=FLAGS.uniprot_database_path,
          use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  num_recycle=FLAGS.num_recycle
  num_ensemble=FLAGS.num_ensemble

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
    if FLAGS.run_model_names and model_name not in FLAGS.run_model_names.split(","):
        continue
    model_config = config.model_config(model_name)
    if run_multimer_system:
      # model_config.model.num_ensemble_eval = num_ensemble
      model_config.model.num_recycle = num_recycle
      model_config.model.num_ensemble_train = num_ensemble
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.common.num_recycle = num_recycle
      model_config.model.num_recycle = num_recycle
      model_config.data.eval.num_ensemble = num_ensemble
    
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  if FLAGS.models_to_relax!="none":
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax)
  else:
    amber_relaxer = None

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax,
        use_custom_templates=FLAGS.use_custom_templates,
        template_alignfile=FLAGS.template_alignfile,
        msa_mode=FLAGS.msa_mode,
        use_custom_MSA_database=FLAGS.use_custom_MSA_database,
        MSA_database=FLAGS.MSA_database,
        save_msa_fasta=FLAGS.save_msa_fasta,
        gen_feats_only=FLAGS.gen_feats_only,
        save_template_names=FLAGS.save_template_names,
        has_gap_chn_brk=FLAGS.has_gap_chn_brk,
        substitute_msa=FLAGS.substitute_msa,
        msa_for_template_query_seq_only=FLAGS.msa_for_template_query_seq_only,
        iptm_interface=FLAGS.iptm_interface,
        feature_prefix=FLAGS.feature_prefix,
        save_ranked_pdb_only=FLAGS.save_ranked_pdb_only,
        save_msa_features_only=FLAGS.save_msa_features_only,
        status_file=FLAGS.status_file)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])

  app.run(main)
