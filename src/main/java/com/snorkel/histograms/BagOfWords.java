package com.snorkel.histograms;

import weka.clusterers.Clusterer;
import weka.core.Instances;

public interface BagOfWords {
	public Clusterer clustering(Instances data,int n);
	public double[] getHistogramImage(Clusterer clusters,Instances imagens);
}
