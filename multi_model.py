import numpy as np
import Bio.PDB
from sklearn.manifold import TSNE

class MultiModel:

    coordinates = [];
    coord_array = [];
    tsne_embedding = [];
    pca_embedding = [];
    umap_embedding = [];
    clusters = [];

    #************************************
    def read_pdbs(self, filenames):

        print("Loading PDB files ...");

        for tmp_file in filenames:
            self.coordinates.append(Bio.PDB.PDBParser().get_structure('hurz', tmp_file)[0]);


    #************************************
    def align_pdbs(self):

        print("Everything aligned to first model...")
        ref_model = self.coordinates[0];
        num_structures = len(self.coordinates);


        for model_ind in range(num_structures):

            alt_model = self.coordinates[model_ind];

            # Build paired lists of c-alpha atoms, ref_atoms and alt_atoms
            ref_atoms = []
            alt_atoms = []
            for (ref_chain, alt_chain) in zip(ref_model, alt_model):
                for ref_res, alt_res in zip(ref_chain, alt_chain):
                    assert ref_res.resname == alt_res.resname
                    assert ref_res.id == alt_res.id

                    # CA = alpha carbon
                    aa = True;
                    try:
                        ref_res['CA'];
                    except:
                        aa = False;

                    if aa:
                        ref_atoms.append(ref_res['CA']);
                        alt_atoms.append(alt_res['CA']);


            # Align these paired atom lists:
            super_imposer = Bio.PDB.Superimposer()
            super_imposer.set_atoms(ref_atoms, alt_atoms)


            if (model_ind == 0):
                # Check for self/self get zero RMS, zero translation
                # and identity matrix for the rotation.
                assert np.abs(super_imposer.rms) < 0.0000001
                assert np.max(np.abs(super_imposer.rotran[1])) < 0.000001
                assert np.max(np.abs(super_imposer.rotran[0]) - np.identity(3)) < 0.000001
            else:
                # Update the structure by moving all the atoms in
                # this model (not just the ones used for the alignment)
                super_imposer.apply(alt_model.get_atoms())

    #*************************************
    def make_vector_representation(self):

        self.coord_array = np.zeros((0,0));
        for tmp_model in self.coordinates:

            tmp_coord = np.zeros((0));
            for tmp_chain in tmp_model:
                for tmp_res in tmp_chain:
                    for tmp_atom in tmp_res:

                        np.append(tmp_coord, tmp_atom.get_vector());

            np.append(self.coord_array, tmp_coord, axis=1)

    #**************************************
    def do_pca_embedding(self):

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2);
        pca.fit(self.coord_array);
        self.pca_embedding = pca.fit_transform;

    #**************************************
    def do_tsne_embedding(self):

        self.tsne_embedding = TSNE(n_components=2).fit_transform(self.coord_array);

    #**************************************
    def do_umap_embedding(self):

        import umap;

        fit = umap.UMAP();
        self.umap_embedding = fit.fit_transform(self.coord_array);

    #***************************************
    def do_clustering(self, num_clusters):

        from sklearn.cluster import KMeans
        self.clusters = KMeans(n_clusters=num_clusters, random_state=0).fit(self.coordinates);

    #***************************************
    def write_pdbs(self):

        num_clusters = self.clusters.cluster_centers.shape[0];
        num_samples = self.clusters.labels.shape[0];

        for tmp_cluster in range(num_clusters):

            #get sample closest to center of cluster
            min_dist = 10^10;
            center_index = 0;
            for tmp_sample in range(num_samples):
                tmp_dist = np.sqrt(np.sum((self.coord_array[tmp_sample,] - self.clusters.cluster_centers[tmp_cluster])^2));

                if tmp_dist < min_dist:
                    center_index = tmp_sample;
                    min_dist = tmp_dist;

            #write the structure
            io = PDBIO()
            io.set_structure(self.coordinates[center_index])
            io.save('Center_Cluster' + repr(tmp_cluster) + '.pdb');

            #write all files in the cluster
