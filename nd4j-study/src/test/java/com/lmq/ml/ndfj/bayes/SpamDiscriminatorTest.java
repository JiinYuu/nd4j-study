package com.lmq.ml.ndfj.bayes;

import org.junit.Test;

import com.lmq.ml.nd4j.bayes.SpamDiscriminator;

public class SpamDiscriminatorTest {

	public @Test void testInit() {
		SpamDiscriminator spamor = new SpamDiscriminator();
		spamor.init();
		spamor.test();
	}
}
