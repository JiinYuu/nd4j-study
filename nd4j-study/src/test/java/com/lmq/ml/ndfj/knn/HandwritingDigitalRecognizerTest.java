package com.lmq.ml.ndfj.knn;

import org.junit.Test;

import com.lmq.ml.nd4j.knn.HandwritingDigitalRecognizer;

public class HandwritingDigitalRecognizerTest {

	public @Test void testInit() {
		new HandwritingDigitalRecognizer().test();
	}
	
	public @Test void testRandom() {
		new HandwritingDigitalRecognizer().testRandom();
	}
}
