package com.snorkel;

import java.io.IOException;

import com.snorkel.description.IFeatureDescription;
import com.snorkel.description.SIFT_Descriptor;
import com.snorkel.detection.IFeatureDetection;
import com.snorkel.detection.SIFT_Detection;
import com.snorkel.helper.Helper;
import com.snorkel.helper.IHelper;
import com.snorkel.histograms.BagOfWords;
import com.snorkel.histograms.K_Means;
import com.snorkel.learning.ILearning;
import com.snorkel.learning.MLP;
import com.snorkel.model.Model;
import com.snorkel.model.ModelImpl;



public class Application {
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		IHelper helper = new Helper();
		ILearning learning = new MLP();
		IFeatureDescription  sift_description= new SIFT_Descriptor();
		IFeatureDetection sift_detection= new SIFT_Detection();
		BagOfWords bagOfWords = new K_Means();
		
		Model model = new Model();
		model.setBoW(bagOfWords);
		model.setF_description(sift_description);
		model.setFeature_detector(sift_detection);
		model.setLearning(learning);		
		
		
		ModelImpl bm = new ModelImpl(model,25);	
		bm.setHelper(helper);	
		bm.Train(learning,bm.ConstructModel("C:/ImagensTeste/train/"));

		
		
	}
}
