package com.lmq.ml.ndfj.logistic;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.lmq.ml.nd4j.logistic.SimpleSample;

import lombok.extern.java.Log;

@Log
public class SimpleSampleTest {

	public @Test void testInit() {
		new SimpleSample().init();
	}
	
	public @Test void testLogistic() {
		SimpleSample ss = new SimpleSample();
		ss.init();
		INDArray weights = ss.logisticRegres();
		log.info(weights.toString());
	}
	
	public @Test void testLogistic2() {
		SimpleSample ss = new SimpleSample();
		ss.init();
		INDArray weights = ss.logisticRegres2();
		log.info(weights.toString());
	}
	
	public @Test void testClassify() {
		SimpleSample ss = new SimpleSample();
		ss.testClassify();
	}
	
	public @Test void testClassify2() {
		SimpleSample ss = new SimpleSample();
		ss.testClassify2();
	}
	
}
