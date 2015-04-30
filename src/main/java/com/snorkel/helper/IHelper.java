package com.snorkel.helper;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Instances;

public interface IHelper {
	public Instances createDataArff(Map<String,Mat> imagens, List<String> classes ) throws IOException;
	public Instances createDataArff(Mat imagem, List<String> classes ) throws IOException;
	public Instances createArffHistogram( HashMap<String, double[]> histograms,List<String> classes) throws IOException;
	public Instances getTrainSet(Instances DataSet) throws IOException;
	public Instances getTestSet(Instances DataSet) throws IOException;
	public void saveCluster(Clusterer cluster, String local) throws IOException;
	public Clusterer loadCluster (String path) throws IOException, ClassNotFoundException;
	public void saveClassifier(Classifier classifier, String local) throws IOException;
	public Classifier loadClassifier (String path) throws IOException, ClassNotFoundException;
	
	public String getFileNameIntermediario();
	

	public void setFileNameIntermediario(String fileNameIntermediario);
	

	public String getFileNameFinal();
	

	public void setFileNameFinal(String fileNameFinal);
	
}
