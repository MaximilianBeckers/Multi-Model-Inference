from MultiModel import MultiModel
import argparse, os, sys
import os.path
import time


# *************************************************************
# ****************** Commandline input ************************
# *************************************************************

cmdl_parser = argparse.ArgumentParser(
	prog=sys.argv[0],
	description='*** Multi-Model analysis ***',
	formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-pdbs', '--pdbs', metavar="model_1.pdb ... model_n.pdb", type=str, required=True,
						 help='Input atomic model files', nargs='+');
cmdl_parser.add_argument('-num_classes', '--num_classes', metavar="4", type=int, required=True,
						 help='number of classes');
cmdl_parser.add_argument('-num_neighbors', '--num_neighbors', metavar="4", type=int, required=False,
						 help='number of neighbors for dimensionality reduction');
cmdl_parser.add_argument('-CA', action='store_true', default=False,
						 help='Flag for limiting analysis to CA atoms');
cmdl_parser.add_argument('-chain', type=str, required=False,
						 help='The chain to be analysed');

# ************************************************************
# ********************** main function ***********************
# ************************************************************

def main():

	start = time.time();

	print('***************************************************');
	print('************** Multi-Model analysis ***************');
	print('***************************************************');

	# get command line input
	args = cmdl_parser.parse_args();

	num_classes = args.num_classes;
	files = args.pdbs;
	if args.chain is None:
		chain = "";
	else:
		chain = args.chain;

	if args.num_neighbors is None:
		num_neighbors = int(0.1*len(files));
	else:
		num_neighbors = args.num_neighbors;

	m = MultiModel();
	m.read_pdbs(files, CA = args.CA, chain=chain);
	m.do_classification(num_classes);
	m.do_pca_embedding();
	m.do_umap_embedding(num_neighbors);
	m.do_tsne_embedding(num_neighbors);
	m.make_plots();

	if args.num_classes>1:
		m.write_pdbs();

	end = time.time();
	totalRuntime = end - start;

	print("****** Summary ******");
	print("Runtime: %.2f" % totalRuntime);



if (__name__ == "__main__"):
	main()
