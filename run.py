from multi_model import MultiModel

files = ["6sae.pdb", "6sag.pdb"];

num_clusters=2;

m = MultiModel();
m.read_pdbs(files);
m.do_clustering(num_clusters)
m.do_pca_embedding();
m.make_plots();
