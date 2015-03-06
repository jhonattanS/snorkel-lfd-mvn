package com.snorkel.histograms;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

public class K_Means implements BagOfWords{

	public Clusterer clustering(Instances data, int n) {
		System.out.println("Gerando Cluster");
		String[] options = new String[1];
		//options[0] = String.valueOf(n);
	    
		SimpleKMeans kmeans  = new SimpleKMeans();
		
		try {
			kmeans.setNumClusters(n);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		kmeans .setDisplayStdDevs(true);
		kmeans .getMaxIterations();
	    try {
	    	kmeans .buildClusterer(data);	    	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    

	    Instances ClusterCenter = kmeans .getClusterCentroids();
	    
	    
	    
	    Instances SDev = kmeans .getClusterStandardDevs();
	    int[] ClusterSize = kmeans .getClusterSizes(); 
	 

	    ClusterEvaluation eval = new ClusterEvaluation();
	    eval.setClusterer(kmeans );
	    try {
			eval.evaluateClusterer(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

//	    for(int i=0;i<ClusterCenter.numInstances();i++){
//	        System.out.println("Cluster#"+( i +1)+ ": "+ClusterSize[i]+" dados .");
//	        System.out.println("Centróide:"+ ClusterCenter.instance(i));
//	        System.out.println("STDDEV:" + SDev.instance(i));
//	        System.out.println("Cluster Evaluation:"+eval.clusterResultsToString());
//
//	    }
	    
	   
	    
		return kmeans ;
	}

	public double[] getHistogramImage(Clusterer clusters, Instances imagens) {
		 		
		double[] histogram = null;
		try {
			histogram = new double[clusters.numberOfClusters()];
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		int i=0;
		for(Instance c: imagens){
			
				try {
					i=clusters.clusterInstance(c);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}					
				histogram[i] = histogram[i]+1;				
		}
		
		
	
		return histogram;	
	}

}
