package com.snorkel.learning;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface ILearning {
	public Classifier train(Instances TrainSet);
	public void evaluator(Classifier classifier, Instances TrainSet, Instances TestSet);
}
