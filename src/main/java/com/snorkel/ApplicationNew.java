package com.snorkel;

import java.io.IOException;

import com.snorkel.description.FREAK_Descriptor;
import com.snorkel.description.IFeatureDescription;
import com.snorkel.description.SIFT_Descriptor;
import com.snorkel.description.SURF_Descriptor;
import com.snorkel.detection.FAST_Detection;
import com.snorkel.detection.Harris_Detection;
import com.snorkel.detection.IFeatureDetection;
import com.snorkel.detection.SIFT_Detection;
import com.snorkel.detection.SURF_Detection;
import com.snorkel.helper.Helper;
import com.snorkel.helper.IHelper;
import com.snorkel.histograms.BagOfWords;
import com.snorkel.histograms.K_Means;
import com.snorkel.learning.ILearning;
import com.snorkel.learning.MLP;
import com.snorkel.model.Model;
import com.snorkel.model.ModelImpl;



public class ApplicationNew {
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		String path ="C:/datasets/TesteIdiota/Train/";
		
		IHelper helper = new Helper();
		ILearning learning = new MLP();
		IFeatureDetection detection= new SURF_Detection();
		IFeatureDescription  description= new SURF_Descriptor();
		
		BagOfWords bagOfWords = new K_Means();
		
		Model model = new Model();
		model.setBoW(bagOfWords);
		model.setF_description(description);
		model.setFeature_detector(detection);
		model.setLearning(learning);		
		
		
		ModelImpl bm = new ModelImpl(model,50);	
		
		helper.setFileNameFinal("FlickrAppleLogoTrain");
		bm.setHelper(helper);	
		
		try {
			bm.ConstructModel(path);
		} catch (Exception e) {
			System.out.println("Verique o caminho");
		}
		
		
		//bm.Train(learning,bm.ConstructModel("C:/datasets/galES/"));

		
		
	}
}
