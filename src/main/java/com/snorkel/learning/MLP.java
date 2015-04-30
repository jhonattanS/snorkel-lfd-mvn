package com.snorkel.learning;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class MLP implements ILearning {

	public Classifier train(Instances TrainSet) {
		Classifier mlp = new MultilayerPerceptron();
		
		try {
			mlp.buildClassifier(TrainSet);
			
			//System.out.println(mlp);
			
		
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mlp;
	}

	public void evaluator(Classifier classifier, Instances TrainSet,
			Instances TestSet) {
		try {
			 Evaluation eval = new Evaluation(TrainSet);
			
			 Random rand = new Random(1);  // using seed = 1
			 int folds = 10;
			 eval.crossValidateModel(classifier, TestSet, folds, rand);
			 System.out.println(eval.toSummaryString());
			 System.out.println(eval.toMatrixString());
			 
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
