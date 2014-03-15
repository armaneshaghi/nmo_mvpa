import nibabel as nb
import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
%pylab inline
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import Bootstrap
from sklearn.utils import shuffle
import pandas as pd

all_gm=nb.load('/home/arman/london_nmo/London/working/\
        mvpa/Mix_gm/all_gm.nii.gz')
nifti_masker = NiftiMasker(mask = '/home/arman/london_nmo/\
        London/working/mvpa/gm/mask/\
        fsl_bin_mask.nii.gz', standardize=True)
all_gm=nifti_masker.fit_transform(all_gm)
df = pd.read_csv('/home/arman/statistical_analysis_with_r_files/\
        london_nmo/london_lesion_load.csv', 
                         index_col=False, header = None)
df2 = pd.read_csv('/home/arman/london_nmo/London/working/mvpa/\
        Mix_gm/list_first_second.txt',
                          index_col = False, header = None)
result = pd.merge(df2, df, on=0, how='outer')

df = pd.read_csv('/home/arman/london_nmo/London/\
        working/mvpa/Mix_gm/list_lesion_load_final.csv',
                    header=None)
lesion_load = df[:][2].values
targets = df[:][1].values
targets = [w.replace('MS', '1') for w in targets]
targets = [w.replace('NMO', '-1') for w in targets]

targets = np.array(targets)

pca = PCA(n_components=4)
pca.fit(all_gm[1:49])

all_gm_red = pca.transform(all_gm)

lesion_load  = lesion_load.reshape(97,1)

n_iter = 1000
n_samples = 97
cv = ShuffleSplit(n_samples, n_iter=n_iter, train_size=49, test_size=48,
            random_state=2)
scores = cross_validation.cross_val_score(
                                                  SVC(C=10, gamma = 1e-5, kernel='rbf'), all_gm_red,
                                                                                            targets,
                                                                                                                                      cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" 
              % (scores.mean(), scores.std()*2))    

gm_wm = np.concatenate((all_gm_red, lesion_load), axis=1)

n_iter = 1000
n_samples = 97
cv = ShuffleSplit(n_samples, n_iter=n_iter, train_size=49, test_size=48,
            random_state=2)
scores = cross_validation.cross_val_score(
                                                  SVC(C=100, gamma = 1e-7, kernel='rbf'), gm_wm,
                                                                                            targets,
                                                                                                                                      cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" 
              % (scores.mean(), scores.std()*2))

