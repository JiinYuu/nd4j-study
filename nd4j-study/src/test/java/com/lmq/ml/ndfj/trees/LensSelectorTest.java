package com.lmq.ml.ndfj.trees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import com.lmq.ml.nd4j.trees.LensSelector;

public class LensSelectorTest {

	public @Test void testInit() {
		//new LensSelector().init();
		double d = 1 / Double.valueOf(2);
		System.out.println(d);
		LensSelector lens = new LensSelector();
		lens.init();
		Object tree = lens.createTree(lens.data, new ArrayList<>(Arrays.asList(lens.labels)));
		System.out.println(tree);
		Map<String, String> sample = new HashMap<>(4);
		sample.put("age", "pre");
		sample.put("prescript", "myope");
		sample.put("astigmatic", "no");
		sample.put("tearRate", "normal");
		String clazz = lens.classify(tree, sample);
		System.out.println(clazz);
	}
}
