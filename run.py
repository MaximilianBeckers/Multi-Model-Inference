from multi_model import MultiModel

files = ["6sae.pdb", "6sag.pdb"];

m = MultiModel();
m.read_pdbs(files);
m.do_pca_embedding();
