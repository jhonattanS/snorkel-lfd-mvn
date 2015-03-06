package com.snorkel.description;

import org.opencv.core.Mat;

import com.snorkel.detection.IFeatureDetection;

public interface IFeatureDescription {
	public Mat getDecriptionOfKeypoints(Mat img, IFeatureDetection f );
	public Mat getDecriptionOfKeypoints(String path_img, IFeatureDetection f );
}
