package com.snorkel.model;

import com.snorkel.description.IFeatureDescription;
import com.snorkel.detection.IFeatureDetection;
import com.snorkel.histograms.BagOfWords;
import com.snorkel.learning.ILearning;



public class Model {

	private ILearning learning;	
	private IFeatureDetection feature_detector;
	private IFeatureDescription f_description;
	private BagOfWords BoW;
	

	
	public Model() {
		// TODO Auto-generated constructor stub
	}



	public ILearning getLearning() {
		return learning;
	}



	public void setLearning(ILearning learning) {
		this.learning = learning;
	}



	public IFeatureDetection getFeature_detector() {
		return feature_detector;
	}



	public void setFeature_detector(IFeatureDetection feature_detector) {
		this.feature_detector = feature_detector;
	}



	public IFeatureDescription getF_description() {
		return f_description;
	}



	public void setF_description(IFeatureDescription f_description) {
		this.f_description = f_description;
	}



	public BagOfWords getBoW() {
		return BoW;
	}



	public void setBoW(BagOfWords boW) {
		BoW = boW;
	}
	
}
