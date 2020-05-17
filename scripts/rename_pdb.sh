#!/bin/bash

for job_id in {201..400}
do
	cd run_$job_id
	pdb=$(ls molProb_*)
	mv $pdb run_${job_id}$pdb
	cd ..
done

mkdir pdbs
mv run_*/run_* pdbs

