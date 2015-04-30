package com.snorkel;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class AvaliacaoDescritores {
	public static void main(String[] args) {
		System.out.println("Iniciando");
		
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		
		FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
	    DescriptorExtractor descriptor = DescriptorExtractor.create(DescriptorExtractor.SIFT);;
	    DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
	   
	  
	    //first image
	    Mat img1 = Highgui.imread("C:/ImagensTeste/boat/img1.pgm");
	    Mat descriptors1 = new Mat();
	    MatOfKeyPoint keypoints1 = new MatOfKeyPoint();

	    detector.detect(img1, keypoints1);
	    descriptor.compute(img1, keypoints1, descriptors1);

	    //second image
	    Mat img2 = Highgui.imread("C:/ImagensTeste/boat/img4.pgm");
	    Mat descriptors2 = new Mat();
	    MatOfKeyPoint keypoints2 = new MatOfKeyPoint();

	    detector.detect(img2, keypoints2);
	    descriptor.compute(img2, keypoints2, descriptors2);


	 
	    MatOfDMatch  matches = new MatOfDMatch();        
	    List<MatOfDMatch>  matchers = new ArrayList<MatOfDMatch>();	    
	    matcher.match(descriptors1,descriptors2,matches);
	   
	    
	    double DIST_LIMIT = 0.15;
	    List<DMatch> matchList = matches.toList();
	    double found_matches = matches.toList().size();
	    List<DMatch> matches_final = new ArrayList<DMatch>();
	    List<DMatch> not_match = new ArrayList<DMatch>();
	    
	    double maior_distancia=0.0;
	    
	    
	    for(int i=0; i<matchList.size(); i++){
	    	
	        if(matchList.get(i).distance >= maior_distancia){
	        	maior_distancia=matchList.get(i).distance;
	        }
	    }
	    
	    for(int i=0; i<matchList.size(); i++){
	    	
	        if(matchList.get(i).distance <= DIST_LIMIT*maior_distancia){
	            matches_final.add(matches.toList().get(i));
	        }else{
	        	not_match.add(matches.toList().get(i));
	        }
	    }
	    
	    System.out.println("\n\n-------Matched---"+matches_final.size()+"-------\n\n");
	    MatOfDMatch matches_final_mat = new MatOfDMatch();
	    matches_final_mat.fromList(matches_final);
	    for(int i=0; i< matches_final.size(); i++){
	       System.out.println(matches_final.get(i)); 
	    }
	    System.out.println("\n\n-------Not matched---"+not_match.size()+"-------\n\n");
	   
	    for(int i=0; i< not_match.size(); i++){
		       System.out.println(not_match.get(i)); 
		 }
	    matchers.add(matches);
	    Mat img3 = new Mat();;
	    
	    Features2d df = new Features2d();
	    double correct_matchs = matches_final.size();
	    System.out.println("Com threshold = "+DIST_LIMIT);
	    System.out.println("Precision: " +correct_matchs +"/"+found_matches+"="+ correct_matchs/found_matches );
	    System.out.println("Recall: " +correct_matchs +"/"+keypoints2.toList().size()+"="+ correct_matchs/keypoints2.toList().size() );
	    
	   // df.drawMatches(img1, keypoints1, img2, keypoints2,   matches, img3);
	    
	    Features2d.drawMatches(img1, keypoints1, img2,keypoints2, new MatOfDMatch(matches_final.toArray(new DMatch[matches_final.size()])), img3, Scalar.all(-1), Scalar.all(-1), new MatOfByte(),   Features2d.NOT_DRAW_SINGLE_POINTS);
	    
	    Highgui.imwrite("saidas/result_match.jpeg", img3); 
	   
	}
}
