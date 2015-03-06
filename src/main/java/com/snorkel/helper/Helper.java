package com.snorkel.helper;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Mat;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Helper implements IHelper {
	double razaoTrainTest=0.7;
	int percentTrain;
	int percentTest;
	
	public Instances createDataArff(Map<String, Mat> imagens,
			List<String> classes) throws IOException {
		Instances  data;
		int numAttributes = 0;	
//		for (Map.Entry<String, Mat> entry : imagens.entrySet()){			
//			numAttributes = entry.getValue().cols();
//			break;
//		}
		List<Mat> values = new ArrayList(imagens.values());
		numAttributes= values.get(0).cols();
		System.out.println("Numero de atributos  "+numAttributes);
		
		//criando atributos
		FastVector  atts = new FastVector();;
		
		for (int i = 0; i < numAttributes; i++) {			
			atts.addElement(new Attribute(""+i+""));			
		}
//		 FastVector fvClassVal = new FastVector(classes.size());
//		 for(String c : classes){
//			 fvClassVal.add(c);
//		 }
//		 Attribute ClassAttribute = new Attribute("class", fvClassVal);
//		 atts.add(ClassAttribute);
//		
//		
//		data.setClassIndex(data.numAttributes()-1);
		data = new Instances("ConjuntoDeImagens",atts, 0);
		
		int index=0;
//		for(int img=0;img<imagens.size();img++){		
//			
//			for(int i = 0 ; i < imagens.get(img).rows();i++){
//				double[] vals = new double[data.numAttributes()];
//				for(int j = 0 ; j < imagens.get(img).cols();j++){
//					vals[j]=imagens.get(img).get(i,j)[0];					
//				}				
//				
//				Instance inst = new DenseInstance(index,vals);		
//				
//				data.add(inst);
//				index++;				
//			}			
//		}		
		for (Map.Entry<String, Mat> entry : imagens.entrySet()){	
			for(int i = 0 ; i < entry.getValue().rows();i++){
				double[] vals = new double[data.numAttributes()];
				for(int j = 0 ; j < entry.getValue().cols();j++){
					vals[j]=entry.getValue().get(i,j)[0];					
				}				
				
				Instance inst = new DenseInstance(index,vals);		
				
				data.add(inst);
				index++;				
			}			
		}
		
		
		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(data);
		 saver.setFile(new File("saidas/test.arff"));		
		 saver.writeBatch();
		
		//System.out.println(data);
		
		return data;
	}

	public Instances createDataArff(Mat imagem, List<String> classes)
			throws IOException {
			Instances  data;
			int numAttributes = imagem.cols();
			//	System.out.println("Numero de atributos  "+numAttributes);
			
			//criando atributos
			FastVector  atts = new FastVector();
			
			for (int i = 0; i < numAttributes; i++) {			
				atts.addElement(new Attribute(""+i+""));			
			}
//			 FastVector fvClassVal = new FastVector(classes.size());
//			 for(String c : classes){
//				 fvClassVal.add(c);
//			 }
//			 Attribute ClassAttribute = new Attribute("class", fvClassVal);
//			 atts.add(ClassAttribute);
//			
//			
//			data.setClassIndex(data.numAttributes()-1);
			data = new Instances("ConjuntoDeImagens",atts, 0);
			int index=0;
				
				
				for(int i = 0 ; i <imagem.rows();i++){
					double[] vals = new double[data.numAttributes()];
					for(int j = 0 ; j < imagem.cols();j++){
						vals[j]=imagem.get(i,j)[0];					
					}				
					
					Instance inst = new DenseInstance(index,vals);		
					
					data.add(inst);
					index++;				
				}			
			
			
			
			 ArffSaver saver = new ArffSaver();
			 saver.setInstances(data);
			 saver.setFile(new File("saidas/test.arff"));		
			 saver.writeBatch();
			
			//System.out.println(data);
			
			return data;
	}

	public Instances createArffHistogram(HashMap<String, double[]> histograms,
			List<String> classes) throws IOException {
		Instances  data ;
		int numAttributes = 0;
	
		List<double[]> values = new ArrayList(histograms.values());
		numAttributes= values.get(0).length;
		//System.out.println("Numero de atributos  "+numAttributes);
		
		//criando atributos
		FastVector  atts = new FastVector();;
		
		for (int i = 0; i < numAttributes; i++) {			
			atts.addElement(new Attribute(""+i+""));			
		}
		
				
		 FastVector fvClassVal = new FastVector(classes.size());
		 for(String c : classes){
			 fvClassVal.add(c);
		 }
		 Attribute ClassAttribute = new Attribute("class", fvClassVal);
		 atts.add(ClassAttribute);
		
		
		
		data = new Instances("ConjuntoDeImagens",atts, 0);
		
		data.setClassIndex(data.numAttributes()-1);
		int index=0;
		
		for (Map.Entry<String,  double[]> entry : histograms.entrySet()){	
			
			double[] vals = new double[data.numAttributes()];
			for(int j = 0 ; j < entry.getValue().length;j++){
				vals[j]=entry.getValue()[j];					
			}				
			
			
			String REGEX = "[a-zA-Z]+";  
			String INPUT = entry.getKey();  			    
	        Matcher matcher = Pattern.compile(REGEX).matcher(INPUT);
	        String s = null;
	        if (matcher.find()) {  	           
	             s = matcher.group();	           
	        }  
			
	        Instance inst = new DenseInstance(1,vals);
	      //  inst.setValue(numAttributes-1, s);
	      //  inst.setClassValue(s);
	        data.add(inst);
	        data.get(index).setClassValue(s);
	        index++;
		}
		
		
		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(data);
		 saver.setFile(new File("saidas/testHist.arff"));		
		 saver.writeBatch();
		
	
		
		return data;
	}

	public Instances getTrainSet(Instances DataSet) throws IOException {
		  Instances data=DataSet;
		  int max = data.size();
		  percentTrain = (int) (max*razaoTrainTest);
		  percentTest = max - percentTrain;		  
		  try{      
		   Instances SetTrain = new Instances(data,0,percentTrain);
		 
		   return SetTrain;
		  }catch(Exception e){
		   System.out.println( e );
		   return null;
		  }
	}

	public Instances getTestSet(Instances DataSet) throws IOException {
		 Instances data=DataSet;
		  int max = data.size();
		  percentTrain = (int) (max*razaoTrainTest);
		  percentTest = max - percentTrain;		  
		  try{      
		   Instances SetTest = new Instances(data,percentTrain,percentTest);
		 
		   return SetTest;
		  }catch(Exception e){
		   System.out.println( e );
		   return null;
		  }
	}

	public void saveCluster(Clusterer cluster, String local) throws IOException {
		FileOutputStream f_out = new FileOutputStream(local);
		// Write object with ObjectOutputStream
		ObjectOutputStream obj_out = new ObjectOutputStream (f_out);
		// Write object out to disk
		obj_out.writeObject ( cluster );
		System.out.println("modelo salvo");
		
	}

	public Clusterer loadCluster(String path) throws IOException,
			ClassNotFoundException {
		// Read from disk using FileInputStream
				FileInputStream f_in = new FileInputStream(path);

				// Read object using ObjectInputStream
				ObjectInputStream obj_in = 	new ObjectInputStream (f_in);

				// Read an object
				Object obj = obj_in.readObject(); 
				System.out.println("modelo carregado");
				return (Clusterer)obj;
	}

	public void saveClassifier(Classifier classifier, String local)
			throws IOException {
		FileOutputStream f_out = new FileOutputStream(local);
		// Write object with ObjectOutputStream
		ObjectOutputStream obj_out = new ObjectOutputStream (f_out);
		// Write object out to disk
		obj_out.writeObject ( classifier );
		System.out.println("modelo salvo");
		
	}

	public Classifier loadClassifier(String path) throws IOException,
		ClassNotFoundException {
		// Read from disk using FileInputStream
		FileInputStream f_in = new FileInputStream(path);

		// Read object using ObjectInputStream
		ObjectInputStream obj_in = 	new ObjectInputStream (f_in);

		// Read an object
		Object obj = obj_in.readObject(); 
		System.out.println("modelo carregado");
		return (Classifier)obj;
	}

}
