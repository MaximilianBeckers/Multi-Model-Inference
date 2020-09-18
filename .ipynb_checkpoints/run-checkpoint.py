from MultiModel import MultiModel
import argparse, sys
import random
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
cmdl_parser.add_argument('-num_classes', '--num_classes', metavar="4", type=int, required=False,
                         help='number of classes');
cmdl_parser.add_argument('-algorithm', '--algorithm', metavar="kmeans", type=str, required=False,
                         help='cluster algorithm');
cmdl_parser.add_argument('-num_neighbors', '--num_neighbors', metavar="1", type=int, required=False,
                         help='number of neighbors for dimensionality reduction');
cmdl_parser.add_argument('-min_cluster_size', '--min_cluster_size', metavar="1", type=int, required=False,
                         help='minimal size of clusters');
cmdl_parser.add_argument('-all_atoms', action='store_true', default=False,
                         help='Flag for performing analysis on all atoms');
cmdl_parser.add_argument('-reduced', action='store_true', default=False,
                         help='Flag for performing classification in reduced space');
cmdl_parser.add_argument('-chain', type=str, required=False,
                         help='The chain to be analysed');


# ************************************************************
# ********************** main function ***********************
# ************************************************************

def main():

    import warnings
    warnings.filterwarnings("ignore")
    
    start = time.time();
    random.seed(2)

    print('***************************************************');
    print('************** Multi-Model analysis ***************');
    print('***************************************************');

    # get command line input
    args = cmdl_parser.parse_args();

    num_classes = args.num_classes;

    files = args.pdbs;

    #for tmv probe
    #files = random.sample(files, 200)
    #files =  files + ["6sae.pdb", "6sag.pdb"];
    if (args.algorithm is None) or (args.algorithm =="kmeans"):
        algorithm = "kmeans";
    elif args.algorithm == "UPGMA":
        algorithm = "UPGMA";
    else:
        print("ERROR: Algorithm not implemented");
        sys.exit();
        

        

    if args.chain is None:
        chain = "";
    else:
        chain = args.chain;

    if args.num_neighbors is None:
        num_neighbors = int(0.1*len(files));
    else:
        num_neighbors = args.num_neighbors;

    if args.min_cluster_size is None:
        min_cluster_size = 10;
    else:
        min_cluster_size = args.min_cluster_size;
        
    m = MultiModel();
    m.read_pdbs(files, CA = not(args.all_atoms), chain=chain);
    m.do_pca_embedding();
    m.do_umap_embedding(num_neighbors);
    m.do_tsne_embedding(num_neighbors);
    m.do_classification(num_classes, min_cluster_size, algorithm, args.reduced);
    m.make_plots();
    
    m.plot_dendrogram();
    #m.probe_tmv();

    if m.good_classes.size > 1:
        #m.SVM();
        m.logistic_regression();
        m.plot_coeffs();
        m.write_pdbs(files, args.reduced);
    else:
        print("Only one class bigger than the threshold. No logistic regression will be performed ...");

    end = time.time();
    totalRuntime = end - start;

    print("****** Summary ******");
    print("Runtime: %.2f" % totalRuntime);

if (__name__ == "__main__"):
    main()