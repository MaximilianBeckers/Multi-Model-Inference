#!/bin/bash

for job_id in {201..400}
do
	mkdir run_$job_id
	cp A_asymm_refine.xml run_$job_id/
	cp 6sae.pdb run_$job_id/
	cp pipeline.sh run_$job_id/
	
	cd run_$job_id
	sbatch pipeline.sh
	cd ..

done
