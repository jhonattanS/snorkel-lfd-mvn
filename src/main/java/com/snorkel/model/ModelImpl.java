package com.snorkel.model;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Mat;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Instances;

import com.snorkel.helper.IHelper;
import com.snorkel.learning.ILearning;

public class ModelImpl implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/*
	 * MODEL VARIABLE
	 */
	
	private Model model;
	
	
	private List<String> classes;	
	
	private File[] images;	
	private int numberOfClusters;	
	
	private IHelper helper;	
	private Clusterer cluster;
	private Classifier classifier;
	
	
	
	
	
	
	
	
	/*-----------------------------CONSTRUCTORS-------------------------------*/
	
	public ModelImpl(Model model, int numberOfClusters) {
		super();
		this.model = model;
		this.numberOfClusters = numberOfClusters;
		this.classes = new ArrayList<String>();	
	}
	
	
	
	/*---------------------------------MODEL--------------------------------*/
	
	public void LoadImages(String folder_diretory){
		File folder = new File(folder_diretory);
		this.images = folder.listFiles();
		
		//EXTRACT THE NAME CLASS FROM IMAGE
		for (File file : images) {
		    
			String REGEX = "[a-zA-Z]+";  
			String INPUT = file.getName();  
			    
	        Matcher matcher = Pattern.compile(REGEX).matcher(INPUT);  
	        if (matcher.find()) {  	           
	            String s = matcher.group();
	            if(!classes.contains(s)){
	            	this.classes.add(s);
	            }
	        }  
		}		
	}
	
	public HashMap<String, Mat> descriptorsImages(String folder_diretory){
		HashMap<String, Mat> descriptors_images = new HashMap<String, Mat>();
		System.out.println("-------------------------------------");
		System.out.println("-----Extraindo descritores-----------");
		System.out.println("-------------------------------------");
		for (File file : images) {			
			System.out.println(folder_diretory + file.getName());
			//extrai os descritores da imagem e joga no map
			Mat k = model.getF_description().getDecriptionOfKeypoints(folder_diretory+file.getName(), model.getFeature_detector());		
			descriptors_images.put(file.getName(), k);
			
		}
		
		return descriptors_images;
	}
	
	//CREATE A .ARFF FILE FOR EACH IMAGE 
	public HashMap<String, Instances> InstancesOfDescriptors(HashMap<String, Mat> descriptors_images) throws IOException{
		HashMap<String, Instances> instancias_description_images = new HashMap<String, Instances>();
		System.out.println("-------------------------------------");
		System.out.println("-----Criando arquivo ARFF------------");
		System.out.println("-------------------------------------");
		
		for (File file : this.images) {			
			if(file!=null){
				System.out.println(file.getName());
				if(descriptors_images!=null){
				Instances data = new Instances(helper.createDataArff(descriptors_images.get(file.getName()), this.classes));
				instancias_description_images.put(file.getName(), data);
				}else{
					System.out.println("ERRO descriptors");
				}
			}else{
				System.out.println("ERRO");
			}
		}
		return instancias_description_images;
	}
	
	public Clusterer createCluster(HashMap<String, Mat> descriptors_images) throws IOException{
		System.out.println("-------------------------------------");
		System.out.println("-----------Criando CLUSTER-----------");
		System.out.println("-------------------------------------");
		Instances data = new Instances(helper.createDataArff(descriptors_images, classes));
		Clusterer nCluster = model.getBoW().clustering(data, numberOfClusters);
		return nCluster;
		
	}
	
	public HashMap<String, double[]> generateHistograms(HashMap<String, Instances> instancias_description_images){
		HashMap<String, double[]> histograms = new HashMap<String, double[]>();
		//a seguir para cada arff das imagens gerar um histograma que é feito clusterizando cada instancia e contando-as
		double[] i = null;		
		for (Map.Entry<String, Instances> entry : instancias_description_images.entrySet())
		{
		    //System.out.println(entry.getKey() + "/" + entry.getValue());
			 i = model.getBoW().getHistogramImage(cluster, entry.getValue());
			 histograms.put(entry.getKey(), i);
		}		
		for (Map.Entry<String, double[]> entry : histograms.entrySet()){
			System.out.println("Imagem "+ entry.getKey());
			for (int j = 0; j < entry.getValue().length; j++) {
				System.out.println("Cluster "+ j+":" + entry.getValue()[j]);
			}
			
		}	
		System.out.println("Arquivos Arff gerados com sucesso!");
		return histograms;
	}
	
	public Instances ConstructModel(String folder_images) throws IOException{
		HashMap<String, double[]> histograms = new HashMap<String, double[]>();
		HashMap<String, Instances> instancias_description_images= new HashMap<String, Instances>();
		LoadImages(folder_images);
		HashMap<String, Mat> descriptors_images = descriptorsImages(folder_images);
		if(descriptors_images!=null){
			instancias_description_images = InstancesOfDescriptors(descriptors_images);
		}
		if(this.cluster==null){
			this.cluster = createCluster(descriptors_images);
			helper.saveCluster(cluster, "models/model.data");
		}else{
			System.out.println("-------------------------------------");
			System.out.println("-----------GERANDO HISTOGRAMAS-----------");
			System.out.println("-------------------------------------");
		}
		
		histograms = generateHistograms(instancias_description_images);		
		return new Instances(helper.createArffHistogram(histograms, classes));
		
	}
	
	
	/*----------------------------LEARNING--------------------------------------*/
	
	public void Train(ILearning learning, Instances Data) throws IOException{
		System.out.println("-------------------------------------");
		System.out.println("---------TREINANDO-------------------");
		System.out.println("-------------------------------------");
		 Instances randomizeData = new Instances(Data);
		 Random rand = new Random(1);
		 randomizeData.randomize(rand);
		 classifier = learning.train(helper.getTrainSet(randomizeData));	
		 
		 learning.evaluator(classifier, helper.getTrainSet(randomizeData), helper.getTestSet(randomizeData));
		 
		 
		
	}
	
	/*-------------------------GETTERS AND SETTERS----------------------------*/
	
	
	public Model getModel() {
		return model;
	}

	public void setModel(Model model) {
		this.model = model;
	}

	public List<String> getClasses() {
		return classes;
	}

	public void setClasses(List<String> classes) {
		this.classes = classes;
	}

	public File[] getImages() {
		return images;
	}

	public void setImages(File[] images) {
		this.images = images;
	}

	public int getNumberOfClusters() {
		return numberOfClusters;
	}

	public void setNumberOfClusters(int numberOfClusters) {
		this.numberOfClusters = numberOfClusters;
	}

	public IHelper getHelper() {
		return helper;
	}

	public void setHelper(IHelper helper) {
		this.helper = helper;
	}

	public Clusterer getCluster() {
		return cluster;
	}

	public void setCluster(Clusterer cluster) {
		this.cluster = cluster;
	}

	public Classifier getClassifier() {
		return classifier;
	}

	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
	}
	
	public void  loadCluster(String path){
		try {
			this.cluster = helper.loadCluster(path);
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
	


	



	


	
	

	

}
