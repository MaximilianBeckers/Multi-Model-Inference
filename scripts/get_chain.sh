#!/bin/bash

chain=A

mkdir "chain_"$chain
for file in *.pdb
do
	fbname=$(basename $file .pdb)
	pdb_out=$fbname"_chain_"${chain}".pdb"	
	echo $pdb_out
	grep -i ' A ' $file > $pdb_out	
	echo "TER" >> $pdb_out
done

mv *_chain_* "chain_"$chain

