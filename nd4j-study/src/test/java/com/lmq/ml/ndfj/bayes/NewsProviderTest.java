package com.lmq.ml.ndfj.bayes;

import org.junit.Test;

import com.lmq.ml.nd4j.bayes.NewsProvider;

public class NewsProviderTest {

	private NewsProvider np = new NewsProvider();
	
	public @Test void testSports() {
//		System.out.println(np.sports());
		System.out.println(np.sports().size());
	}
	
	public @Test void testFinance() {
//		System.out.println(np.finance());
		System.out.println(np.finance().size());
	}
}
