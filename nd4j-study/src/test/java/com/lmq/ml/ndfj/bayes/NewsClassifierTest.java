package com.lmq.ml.ndfj.bayes;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;

import com.lmq.ml.nd4j.bayes.NewsClassifier;

public class NewsClassifierTest {

	public @Test void testInit() throws Exception {
		double sum = 0.0;
		double sum2 = 0.0;
		for(int i = 0; i < 20; i++) {
			NewsClassifier classifier = new NewsClassifier();
			classifier.init();
			Pair<Double, Double> rate = classifier.test();
			sum += rate.getFirst();
			sum2 += rate.getSecond();
		}
		System.out.println("----平均错误率：" + (sum * 100 / 20) + "%");
		System.out.println("归一化平均错误率：" + (sum2 * 100 / 20) + "%");
	}
}
