package com.snorkel.detection;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Harris_Detection implements IFeatureDetection{

	public MatOfKeyPoint getKeyPoints(String caminho_imagem) {
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		
		Mat img = Highgui.imread(caminho_imagem);		
		Imgproc processar = new Imgproc();
		
		processar.cvtColor(img, img, processar.COLOR_BGR2GRAY);
		
		MatOfKeyPoint keyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector=FeatureDetector.create(FeatureDetector.HARRIS);
		
		featureDetector.detect(img, keyPoints);	
		
		return keyPoints;
	}

	public MatOfKeyPoint getKeyPoints(Mat imagem) {
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		Mat img = new Mat();
		imagem.copyTo(img);
		
		MatOfKeyPoint keyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector=FeatureDetector.create(FeatureDetector.HARRIS);
		
		featureDetector.detect(img, keyPoints);	
		
		return keyPoints;
	}

}
