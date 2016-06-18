import numpy as np
from sklearn.decomposition import PCA

# extract features from a sliding window of frames
def extractFeatures_by_sliding_window(data):
	featureWindow = 25
	padding = np.array([[0] * 32] * featureWindow)
	data = np.vstack((padding,np.vstack((data, padding))))
	features = []
	for i in range(featureWindow, data.shape[0] - featureWindow):
		features.append(data[i-featureWindow:i+featureWindow].flatten())
	return np.array(features)

def extractFeatures_others(data):
	return data


def extractFeatures_PCA(data, numPCs):
    pca = RandomizedPCA(n_components=numPCs)
    return pca.fit_transform(data) 