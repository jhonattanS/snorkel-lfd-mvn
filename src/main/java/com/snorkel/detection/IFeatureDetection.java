package com.snorkel.detection;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public interface IFeatureDetection {
	public MatOfKeyPoint getKeyPoints(String caminho_imagem);
	public MatOfKeyPoint getKeyPoints(Mat imagem);
}
