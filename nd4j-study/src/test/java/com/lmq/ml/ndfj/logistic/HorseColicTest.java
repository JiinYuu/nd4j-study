package com.lmq.ml.ndfj.logistic;

import org.junit.Test;

import com.lmq.ml.nd4j.logistic.HorseColic;

public class HorseColicTest {

	public @Test void testInit() {
		new HorseColic().init();
	}
	
	public @Test void testLogisticRegression() {
		HorseColic hc = new HorseColic();
		hc.init();
		System.out.println(hc.logisticRegres());
		System.out.println(hc.logisticRegres2());
	}
	
	public @Test void testTest() {
		HorseColic hc = new HorseColic();
		hc.test();
	}
	
	public @Test void testTest2() {
		HorseColic hc = new HorseColic();
		hc.test2();
	}
}
