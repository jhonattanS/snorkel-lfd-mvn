package com.snorkel.description;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.highgui.Highgui;

import com.snorkel.detection.IFeatureDetection;

public class SIFT_Descriptor implements IFeatureDescription {

	public Mat getDecriptionOfKeypoints(Mat img, IFeatureDetection f) {
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);		
		Mat descriptors= new Mat();
		descriptorExtractor.compute(img, f.getKeyPoints(img), descriptors);		
		return descriptors;
	}

	public Mat getDecriptionOfKeypoints(String path_img, IFeatureDetection f) {
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		DescriptorExtractor descriptorExtractor;
		descriptorExtractor= DescriptorExtractor.create(DescriptorExtractor.SIFT);	
		Mat img = Highgui.imread(path_img);
		//System.out.println("leu");
		Mat descriptors= new Mat();
		descriptorExtractor.compute(img, f.getKeyPoints(img), descriptors);		
		//System.out.println("computou");
		//System.out.println(descriptors.dump());
		return descriptors;
	}

}
