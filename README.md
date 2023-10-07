## TCRmodel2
TCRmodel2 is a deep learning algorithm, based on AlphaFold, to model TCR-pMHC complex structures, as well as unbound TCR structures. 

While we offer the option to download and install TCRmodel2 locally via this repository, we recommend considering our web server for generating predictions. The web server offers a user-friendly interface and eliminates the need for local installations. You can access the web server at the following URL: 

https://tcrmodel.ibbr.umd.edu/

However, if you would like to generate large sets of models, local installation is recommended, to avoid overloading the web server. 

If you use TCRmodel2, please cite:

Yin R, Ribeiro-Filho HV, Lin V, Gowthaman R, Cheung M, Pierce BG. (2023) TCRmodel2: high-resolution modeling of T cell receptor recognition using deep learning. Nucleic Acids Res. In press. https://doi.org/10.1093/nar/gkad356

## Table of contents
- [Quick start](#quick-start)
- [Generate TCR-pMHC complex predictions](#TCRpMHC-predictions)
- [Generate unbound TCR predictions](#TCR-predictions)
- [Thanks](#thanks)
- [References](#references)
- [Copyright and license](#copyright-and-license)

## Quick start
The TCRmodel2 code is adapted from AlphaFold v.2.3.0. 

### Requirements
NVIDIA cuda driver >= 11.2

### Install softwares
To install the dependencies, we recommend the following steps:
1. Install alphaFold requirements in a conda environment. Here's a useful resource if you prefer to install AlphaFold without Docker: https://github.com/kalininalab/alphafold_non_docker 
2. Install additional packages: [ANARCI](https://github.com/oxpig/ANARCI) and [MDAnalysis](https://www.mdanalysis.org/pages/installation_quick_start/) to the conda environment created from previous step. These two packages are not required for generating structural predictions. ANARCI is used to trim TCR to variable domains only, and for renumbering PDB outputs. MDAnalysis is used for output renumbering and output alignment.

``` bash
conda install -c bioconda anarci
conda config --add channels conda-forge
conda install mdanalysis
``` 

### Download database
While the majority of database files can be found in data/dabases/ folder, due to file size limit, one would need to: 
1. unzip pdb sequence database file:
```bash
cd data/databases
tar -xvzf pdb_seqres.txt.tar.gz
```
2. download pdb_mmcif and params database (around 120 GB total after unzip) used by alphafold to a database folder of your choice, the path of which will be pass as a ori_db variable to the run_tcrmodel2.py and run_tcrmodel2_ub_tcr.py script. Please refer to the download instructions in [download_pdb_mmcif.sh](https://github.com/deepmind/alphafold/blob/18e12d61314214c51ca266d192aad3cc6619018a/scripts/download_pdb_mmcif.sh) and [download_alphafold_params.sh](https://github.com/deepmind/alphafold/blob/6a3af1adb3bbbc53562da100e3819b2fc882f915/scripts/download_alphafold_params.sh) in [alphafold](https://github.com/deepmind/alphafold/) repository. 

## Generate TCR-pMHC complex predictions
Workflow for creating TCR-pMHC complex structure predictions:
1. Receive TCR alpha, beta, peptide, MHC sequences
2. Build pMHC template alignment file
3. Generate MSA features using a reduced database for all chains, considered seperatedly
4. Generate all other features by concatenating peptide MHC into one chain
5. Predict structures
6. Output 5 structures and a text file containing 1) templates used 2) prediction scores 

Peptide length requirement:
* For class I TCR-pMHC complexes, kindly ensure that the peptide length is between 8-15.
* For class II TCR-pMHC complexes, kindly ensure that the peptide input is 11 aa in length. Specifically, it should consist of a 9 aa core with an additional 1 aa at both the N-terminal and C-terminal of the core peptide.

To make a class I TCR-pMHC prediction:
``` bash
python run_tcrmodel2.py \
--job_id=test_clsI_6kzw \
--output_dir=experiments/ \
--tcra_seq=AQEVTQIPAALSVPEGENLVLNCSFTDSAIYNLQWFRQDPGKGLTSLLLIQSSQREQTSGRLNASLDKSSGRSTLYIAASQPGDSATYLCAVTNQAGTALIFGKGTTLSVSS \
--tcrb_seq=NAGVTQTPKFQVLKTGQSMTLQCSQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSIRGSRGEQFFGPGTRLTVL \
--pep_seq=RLPAKAPLL \
--mhca_seq=SHSLKYFHTSVSRPGRGEPRFISVGYVDDTQFVRFDNDAASPRMVPRAPWMEQEGSEYWDRETRSARDTAQIFRVNLRTLRGYYNQSEAGSHTLQWMHGCELGPDGRFLRGYEQFAYDGKDYLTLNEDLRSWTAVDTAAQISEQKSNDASEAEHQRAYLEDTCVEWLHKYLEKGKETLLH \
--ori_db=/path/to/alphafold_database #set it as the path to the folder containing pdb_mmcif and params
```

To make a class II TCR-pMHC prediction:
``` bash
python run_tcrmodel2.py \
--job_id=test_clsII_7t2c \
--output_dir=experiments \
--tcra_seq=LAKTTQPISMDSYEGQEVNITCSHNNIATNDYITWYQQFPSQGPRFIIQGYKTKVTNEVASLFIPADRKSSTLSLPRVSLSDTAVYYCLVGDTGFQKLVFGTGTRLLVSP \
--tcrb_seq=GAVVSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSARDPGGGGSSYEQYFGPGTRLTVT \
--pep_seq=LAWEWWRTV \
--mhca_seq=IKADHVSTYAAFVQTHRPTGEFMFEFDEDEMFYVDLDKKETVWHLEEFGQAFSFEAQGGLANIAILNNNLNTLIQRSNHTQAT \
--mhcb_seq=PENYLFQGRQECYAFNGTQRFLERYIYNREEFARFDSDVGEFRAVTELGRPAAEYWNSQKDILEEKRAVPDRMCRHNYELGGPMTLQR \
--ori_db=/path/to/alphafold_database #set it as the path to the folder containing pdb_mmcif and params
```

You may use additional flags in run_tcrmodel2.py to control additional behaviors of the script. To see a list of flags:
``` bash
python run_tcrmodel2.py --help
```

## Generate unbound TCR predictions
Workflow for creating TCR-pMHC complex structure predictions:
1. Receive TCR alpha, beta sequences
2. Generate MSA features using reduced database, and modified TCR template search protocol. 
3. Predict structures
4. Output 5 structures and a text file containing 1) templates used 2) prediction scores 

To make a unbound TCR prediction:
``` bash
python run_tcrmodel2_ub_tcr.py \
--job_id=test_tcr_7t2b \
--output_dir=experiments \
--tcra_seq=SQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGRGLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCATDKKGGATNKLIFGTGTLLAVQP \
--tcrb_seq=NAGVTQTPKFRVLKTGQSMTLLCAQDMNHEYMYWYRQDPGMGLRLIHYSVGEGTTAKGEVPDGYNVSRLKKQNFLLGLESAAPSQTSVYFCASSQGGGEQYFGPGTRLTVT \
--ori_db=/path/to/alphafold_database #set it as the path to the folder containing pdb_mmcif and params
```

You may use additional flags in run_tcrmodel2_ub_tcr.py to control additional behaviors of the script. To see a list of flags:
``` bash
python run_tcrmodel2_ub_tcr.py --help
```


## Thanks
We would like to thank [alphafold](https://github.com/deepmind/alphafold/), [alphafold_finetune](https://github.com/phbradley/alphafold_finetune), [ColabFold](https://github.com/sokrypton/ColabFold) teams for developing and distributing the code. The content inside alphafold/ folder is modified from [alphafold/](https://github.com/deepmind/alphafold/tree/main/alphafold) of [alphafold](https://github.com/deepmind/alphafold/) repository. The featurization of custom template is modified from [predict_utils.py](https://github.com/phbradley/alphafold_finetune/blob/main/predict_utils.py) of [alphafold_finetune](https://github.com/phbradley/alphafold_finetune). Chain break introduction, as well as making mock template feature steps are modified from [batch.py](https://github.com/sokrypton/ColabFold/blob/aa7284b56c7c6ce44e252787011a6fd8d2817f85/colabfold/batch.py) of [ColabFold](https://github.com/sokrypton/ColabFold).

## Reference
Yin R, Ribeiro-Filho HV, Lin V, Gowthaman R, Cheung M, Pierce BG. (2023) TCRmodel2: high-resolution modeling of T cell receptor recognition using deep learning. Nucleic Acids Res. In press. https://doi.org/10.1093/nar/gkad356

## Copyright and license
Apache License 2.0
