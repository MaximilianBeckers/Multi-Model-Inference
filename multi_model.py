import numpy as np
import Bio.PDB
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class MultiModel:

    coordinates = [];
    coord_array = [];
    tsne_embedding = [];
    pca_embedding = [];
    explained_variances = [];
    umap_embedding = [];
    classes = [];


    #************************************
    def read_pdbs(self, filenames, align=True, CA=False):

        print("Loading PDB files ...");

        for tmp_file in filenames:
            self.coordinates.append(Bio.PDB.PDBParser().get_structure('hurz', tmp_file)[0]);

        #align the structures
        if align:
            print("Aligning the atomic models based on C-alpha positions ...")
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


        #make vector representation of coordinates
        self.coord_array = [];
        for tmp_model in self.coordinates:

            tmp_coord = [];
            for tmp_chain in tmp_model:
                for tmp_res in tmp_chain:
                    if(CA):
                        try:#if CA atom exists, append it
                            tmp_atom = tmp_res['CA'];
                            tmp_coord.append(tmp_atom.get_coord());
                        except:
                            pass
                    else:
                        #append all atoms
                        for tmp_atom in tmp_res:
                            tmp_coord.append( tmp_atom.get_coord());

            tmp_coord = np.asarray(tmp_coord).flatten();
            self.coord_array.append( tmp_coord);

        self.coord_array = np.asarray(self.coord_array)


    #**************************************
    def do_pca_embedding(self):

        from sklearn.decomposition import PCA
        pca = PCA();
        pca.fit(self.coord_array);
        self.explained_variances = pca.explained_variance_ratio_ * 100;

        pca = PCA(n_components=2);
        pca.fit(self.coord_array);
        self.pca_embedding = pca.fit_transform(self.coord_array);


    #**************************************
    def do_tsne_embedding(self):

        self.tsne_embedding = TSNE(n_components=2).fit_transform(self.coord_array);


    #**************************************
    def do_umap_embedding(self):

        import umap;

        fit = umap.UMAP();
        self.umap_embedding = fit.fit_transform(self.coord_array);


    #***************************************
    def do_classification(self, num_classes):

        from sklearn.cluster import KMeans
        self.classes = KMeans(n_clusters=num_classes, random_state=0).fit(self.coord_array);


    #***************************************
    def make_plots(self):

        #plot pca
        plt.scatter(self.pca_embedding[:,0], self.pca_embedding[:,1], c=self.classes.labels_)
        plt.title('PCA plot');
        plt.show()

        #plot explained variances
        plt.plot(range(1,21,1), self.explained_variances[0:20], linewidth=1.5);
        plt.xticks(np.arange(1, 21, step=1))
        plt.xlabel('Principal Component')
        plt.ylabel('Explained variance [%]')
        plt.title('Explained variances per principal component');
        plt.show();


    #***************************************
    def write_pdbs(self):

        num_classes = self.classes.cluster_centers_.shape[0];
        num_samples = self.classes.labels.shape[0];

        for tmp_class in range(num_classes):

            #get sample closest to center of cluster
            min_dist = 10^10;
            center_index = 0;
            for tmp_sample in range(num_samples):
                tmp_dist = np.sqrt(np.sum((self.coord_array[tmp_sample,] - self.classes.cluster_centers[tmp_class])^2));

                if tmp_dist < min_dist:
                    center_index = tmp_sample;
                    min_dist = tmp_dist;

            #write the structure
            io = PDBIO()
            io.set_structure(self.coordinates[center_index])
            io.save('Center_Cluster' + repr(tmp_class) + '.pdb');

            #write all files in the cluster
