#!/bin/bash
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                      # number of cores
#SBATCH --partition=htc
#SBATCH --mem 10G                  # memory pool for all cores
#SBATCH -t 2-0:00                   # runtime limit (D-HH:MM:SS)

##########################
###### define input ######
##########################
halfmap1=EMD-3061-half1.mrc
halfmap2=EMD-3061-half2.mrc
map=../emd_10129.map
pdb_input=6sae.pdb
res=1.9
##########################

###### start rosetta ######
module load Rosetta

/g/easybuild/x86_64/CentOS/7/haswell/software/Rosetta/2020.03.61102-foss-2018b/bin/rosetta_scripts.mpi.linuxgccrelease \
-database /g/easybuild/x86_64/CentOS/7/haswell/software/Rosetta/2020.03.61102-foss-2018b/database/ \
-in::file::s $pdb_input \
-parser::protocol A_asymm_refine.xml \
-parser::script_vars denswt=35 rms=1.5 reso=$res map=$map \
-ignore_unrecognized_res \
-edensity::mapreso $res \
-default_max_cycles 200 \
-edensity::cryoem_scatterers \
-beta \
-out::suffix _asymm \
-crystal_refine

fbname=$(basename $pdb_input .pdb)
pdb_rosetta=$fbname"_asymm_0001.pdb"

###### start phenix ######
module load phenix

#set some internal names
fbname=$(basename $pdb_rosetta .pdb)
refinedPDB=$fbname"_real_space_refined.pdb"

#run phenix
phenix.real_space_refine $pdb_rosetta $map resolution=$res run=minimization_global+local_grid_search

#run molprobity
phenix.molprobity $refinedPDB

#place molprobity in filename
molprob=$(grep 'MolProbity score' molprobity.out | tail -c 5)
new_file_name="molProb_"$molprob".pdb"
mv $refinedPDB $new_file_name

