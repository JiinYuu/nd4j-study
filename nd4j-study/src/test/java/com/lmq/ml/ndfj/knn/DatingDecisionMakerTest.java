package com.lmq.ml.ndfj.knn;

import org.junit.Test;

import com.lmq.ml.nd4j.knn.DatingDecisionMaker;

public class DatingDecisionMakerTest {

	public @Test void testInit() {
		new DatingDecisionMaker().init();
	}
	
	public @Test void testTest() {
		
		DatingDecisionMaker dsm = new DatingDecisionMaker();
		double sum = 0.0;
		for(int i = 0; i < 20; i++) {
			dsm.init();
			double rate = dsm.test(1);
			sum += rate;
		}
		System.out.println("平均错误率：" + (sum / 20));
	}
	
	public @Test void testTest2() {
		
		DatingDecisionMaker dsm = new DatingDecisionMaker();
		double sum = 0.0;
		for(int i = 0; i < 20; i++) {
			dsm.init();
			double rate = dsm.test2(4);
			sum += rate;
		}
		System.out.println("平均错误率：" + (sum / 20));
	}
}
