package com.lmq.ml.nd4j.bayes;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lmq.ml.nd4j.common.Stemmer;


public class SpamDiscriminator {

	private static final String TRAINING_DATA_DIR = "bayes/email";
	private INDArray data;
	private INDArray label;
	private INDArray p1, p0;
	
	private INDArray testData;
	private INDArray testLabel;

	public void init() {
		Random random = new Random();
		File file = new File(
				SpamDiscriminator.class.getClassLoader().getResource(TRAINING_DATA_DIR).getFile());
		Stemmer stemmer = new Stemmer();
		Set<String> wordsSet = stemmer.cutToSetFromFile(file);
		String[] words = wordsSet.toArray(new String[wordsSet.size()]);
		File spam = new File(SpamDiscriminator.class.getClassLoader().getResource(TRAINING_DATA_DIR + "/spam").getFile());
		File ham = new File(SpamDiscriminator.class.getClassLoader().getResource(TRAINING_DATA_DIR + "/ham").getFile());
		File[] spams = spam.listFiles();
		File[] hams = ham.listFiles();
		List<File> spamsList = new ArrayList<>(Arrays.asList(spams));
		List<File> hamsList = new ArrayList<>(Arrays.asList(hams));
		
		Set<Integer> testSpams = new HashSet<>(5);
		Set<Integer> testHams = new HashSet<>(5);
		while(true) {
			int rand = random.nextInt(spams.length);
			if(testSpams.size() < 5) {
				testSpams.add(rand);
			}
			int rand2 = random.nextInt(hams.length);
			if(testHams.size() < 5) {
				testHams.add(rand2);
			}
			if(testHams.size() >= 5 && testSpams.size() >= 5) {
				break;
			}
		}
		
		this.testData = Nd4j.zeros(10, words.length);
		this.testLabel = Nd4j.zeros(10, 1);
		
		Iterator<Integer> it = testSpams.iterator();
		File[] spamsTest = new File[5];
		int testI = 0;
		for(int i = 0; i < spamsTest.length; i++) {
			int index = it.next();
			spamsTest[i] = spams[index];
			Map<String, List<String>> wordsCount = stemmer.cutToListFromFile(spamsTest[i]).stream().collect(Collectors.groupingBy(String::toString));
			int j = 0;
			for(String word : words) {
				if(wordsCount.containsKey(word)) {
					this.testData.put(testI, j, wordsCount.get(word).size());
				}
				j++;
			}
			this.testLabel.put(testI, 0, 1);
			testI++;
			
		}
		spamsList.removeAll(Arrays.asList(spamsTest));
		
		Iterator<Integer> it2 = testHams.iterator();
		File[] hamsTest = new File[5];
		for(int i = 0; i < hamsTest.length; i++) {
			int index = it2.next();
			hamsTest[i] = hams[index];
			Map<String, List<String>> wordsCount = stemmer.cutToListFromFile(hamsTest[i]).stream().collect(Collectors.groupingBy(String::toString));
			int j = 0;
			for(String word : words) {
				if(wordsCount.containsKey(word)) {
					this.testData.put(testI, j, wordsCount.get(word).size());
				}
				j++;
			}
			this.testLabel.put(testI, 0, 0);
			testI++;
		}
		hamsList.removeAll(Arrays.asList(hamsTest));
		
		spams = spamsList.toArray(new File[spamsList.size()]);
		hams = hamsList.toArray(new File[hamsList.size()]);
		
		this.data = Nd4j.zeros(spams.length + hams.length, words.length);
		this.label = Nd4j.zeros(spams.length + hams.length, 1);
		int i = 0;
		for(File f : spams) {
			Map<String, List<String>> wordsCount = stemmer.cutToListFromFile(f).stream().collect(Collectors.groupingBy(String::toString));
			int j = 0;
			for(String word : words) {
				if(wordsCount.containsKey(word)) {
					this.data.put(i, j, wordsCount.get(word).size());
				}
				j++;
			}
			this.label.put(i, 0, 1);
			i++;
		}
		
		INDArray p1sum = Nd4j.ones(1, words.length).add(this.data.sum(0));
		INDArray p1sumAll = p1sum.sum(1);
		this.p1 = p1sum.div(p1sumAll);
		for(File f : hams) {
			Map<String, List<String>> wordsCount = stemmer.cutToListFromFile(f).stream().collect(Collectors.groupingBy(String::toString));
			int j = 0;
			for(String word : words) {
				if(wordsCount.containsKey(word)) {
					this.data.put(i, j, wordsCount.get(word).size());
				}
				j++;
			}
			this.label.put(i, 0, 0);
			i++;
		}
		INDArray p0sum = Nd4j.ones(1, words.length).add(this.data.sum(0).sub(p1sum.sub(1)));
		INDArray p0sumAll = p0sum.sum(1);
		this.p0 = p0sum.div(p0sumAll);
	}
	
	public void test() {
		int errorCount = 0;
		for(int i = 0; i < 10; i++) {
			INDArray test = this.testData.getRow(i);
			double p1 = test.mul(Transforms.log(this.p1, 2)).sum(1).getDouble(0, 0);
			double p0 = test.mul(Transforms.log(this.p0, 2)).sum(1).getDouble(0, 0);
			
			int label = p1 > p0 ? 1 : 0;
			int tlabel = this.testLabel.getInt(i, 0);
			if(label != tlabel) {
				errorCount++;
			}
			// test.lo	
		}
		System.out.println("错误数：" + errorCount);
	}

}
