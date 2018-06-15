package com.lmq.ml.nd4j.knn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.util.StringUtils;

public class HandwritingDigitalRecognizer {

	public static final String TRAINING_DATA_DIR = "knn/trainingDigits";
	
	public static final String TEST_DATA_DIR = "knn/testDigits";
	
	private INDArray data, label;
	
	private INDArray zero, one, two, three, four, five, six, seven, eight, nine;
	
	public void init() {
		File trainingDigitsDir = new File(HandwritingDigitalRecognizer.class.getClassLoader().getResource(TRAINING_DATA_DIR).getFile());
		File[] digitFiles =  trainingDigitsDir.listFiles();
		data = Nd4j.zeros(digitFiles.length, 1024);
		label = Nd4j.zeros(digitFiles.length, 1);
//		this.zero = Nd4j.zeros(1, 1024);
//		this.one = Nd4j.zeros(1, 1024);
//		this.two = Nd4j.zeros(1, 1024);
//		this.three = Nd4j.zeros(1, 1024);
//		this.four = Nd4j.zeros(1, 1024);
//		this.five = Nd4j.zeros(1, 1024);
//		this.six = Nd4j.zeros(1, 1024);
//		this.seven = Nd4j.zeros(1, 1024);
//		this.eight = Nd4j.zeros(1, 1024);
//		this.nine = Nd4j.zeros(1, 1024);
		for(int i = 0; i < digitFiles.length; i++) {
			File digitFile = digitFiles[i];
			String name = digitFile.getName();
			int curLabel = Integer.valueOf(name.split("\\.")[0].split("_")[0]);
			switch(curLabel) {
			case 0:
				if(this.zero == null) {
					this.zero = Nd4j.zeros(1, 1024);
				} else {
					this.zero = Nd4j.vstack(this.zero, Nd4j.zeros(1, 1024));
				}
				break;
			case 1:
				if(this.one == null) {
					this.one = Nd4j.zeros(1, 1024);
				} else {
					this.one = Nd4j.vstack(this.one, Nd4j.zeros(1, 1024));
				}
				break;
			case 2:
				if(this.two == null) {
					this.two = Nd4j.zeros(1, 1024);
				} else {
					this.two = Nd4j.vstack(this.two, Nd4j.zeros(1, 1024));
				}
				break;
			case 3:
				if(this.three == null) {
					this.three = Nd4j.zeros(1, 1024);
				} else {
					this.three = Nd4j.vstack(this.three, Nd4j.zeros(1, 1024));
				}
				break;
			case 4:
				if(this.four == null) {
					this.four = Nd4j.zeros(1, 1024);
				} else {
					this.four = Nd4j.vstack(this.four, Nd4j.zeros(1, 1024));
				}
				break;
			case 5:
				if(this.five == null) {
					this.five = Nd4j.zeros(1, 1024);
				} else {
					this.five = Nd4j.vstack(this.five, Nd4j.zeros(1, 1024));
				}
				break;
			case 6:
				if(this.six == null) {
					this.six = Nd4j.zeros(1, 1024);
				} else {
					this.six = Nd4j.vstack(this.six, Nd4j.zeros(1, 1024));
				}
				break;
			case 7:
				if(this.seven == null) {
					this.seven = Nd4j.zeros(1, 1024);
				} else {
					this.seven = Nd4j.vstack(this.seven, Nd4j.zeros(1, 1024));
				}
				break;
			case 8:
				if(this.eight == null) {
					this.eight = Nd4j.zeros(1, 1024);
				} else {
					this.eight = Nd4j.vstack(this.eight, Nd4j.zeros(1, 1024));
				}
				break;
			case 9:
				if(this.nine == null) {
					this.nine = Nd4j.zeros(1, 1024);
				} else {
					this.nine = Nd4j.vstack(this.nine, Nd4j.zeros(1, 1024));
				}
				break;
			}
			label.put(i, 0, curLabel);
			try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(digitFile)))) {
				
				String line = null;
				int j = 0;
				while(StringUtils.hasText((line = reader.readLine()))) {
					String[] lineArr = line.split("");
					for(String s : lineArr) {
						data.put(i, j, Integer.valueOf(s));
						switch(curLabel) {
						case 0:
							this.zero.put(this.zero.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 1:
							this.one.put(this.one.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 2:
							this.two.put(this.two.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 3:
							this.three.put(this.three.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 4:
							this.four.put(this.four.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 5:
							this.five.put(this.five.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 6:
							this.six.put(this.six.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 7:
							this.seven.put(this.seven.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 8:
							this.eight.put(this.eight.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						case 9:
							this.nine.put(this.nine.shape()[0] - 1, j, Integer.valueOf(s));
							break;
						}
						j++;
					}
				}
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
		
	}
	
	public void test() {
		this.init();
		File testDigitsDir = new File(HandwritingDigitalRecognizer.class.getClassLoader().getResource(TEST_DATA_DIR).getFile());
		File[] testFiles =  testDigitsDir.listFiles();
		double errorCount = 0;
		for(File testFile : testFiles) {
			String name = testFile.getName();
			int curLabel = Integer.valueOf(name.split("\\.")[0].split("_")[0]);
			try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(testFile)))) {
				INDArray testData = Nd4j.zeros(1, 1024);
				String line = null;
				int j = 0;
				while(StringUtils.hasText((line = reader.readLine()))) {
					String[] lineArr = line.split("");
					for(String s : lineArr) {
						if("0".equals(s)) {
							j++;
							continue;
						}
						testData.put(0, j++, Integer.valueOf(s));
					}
				}
				int dataSize = data.shape()[0];
				List<Pair<Integer, Double>> indexAndDistance = new ArrayList<>(dataSize);
				for(int i = 0; i < dataSize; i++) {
					INDArray row = data.getRow(i);
					double dis = row.distance2(testData);
					indexAndDistance.add(new Pair<>(i, dis));
				}
				Collections.sort(indexAndDistance, (p1, p2) -> p1.getValue().compareTo(p2.getValue()));
				Map<Integer, Pair<Integer, Integer>> testLabels = new HashMap<>(5);
				for(int i = 0; i < 1; i++) {
					int testLabel = label.getInt(indexAndDistance.get(i).getKey(), 0);
					if(testLabels.containsKey(testLabel)) {
						Pair<Integer, Integer> p = testLabels.get(testLabel);
						
						testLabels.put(testLabel, new Pair<>(testLabel, p.getValue() + 1));
					} else {
						testLabels.put(testLabel, new Pair<>(testLabel, 1));
					}
				}
				
				List<Pair<Integer, Integer>> testLabelList = testLabels.values().stream().sorted((p1, p2) -> p2.getValue().compareTo(p1.getValue())).collect(Collectors.toList());
				int testLabel = testLabelList.get(0).getKey();
				
				if(curLabel != testLabel) {
					errorCount++;
					System.out.println("正确数字：" + curLabel + "，识别为：" + testLabel + "，文件名：" + testFile.getName());
				}
				
				
			} catch(Exception e) {
				
			}
		}
		System.out.println("错误数：" + errorCount);
		System.out.println("错误率：" + (errorCount * 100 / testFiles.length) + "%");
	}
	
	public void testRandom() {
		this.init();
		File testDigitsDir = new File(HandwritingDigitalRecognizer.class.getClassLoader().getResource(TEST_DATA_DIR).getFile());
		File[] testFiles =  testDigitsDir.listFiles();
		double errorCount = 0;
		for(File testFile : testFiles) {
			String name = testFile.getName();
			int curLabel = Integer.valueOf(name.split("\\.")[0].split("_")[0]);
			try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(testFile)))) {
				INDArray testData = Nd4j.zeros(1, 1024);
				String line = null;
				int j = 0;
				while(StringUtils.hasText((line = reader.readLine()))) {
					String[] lineArr = line.split("");
					for(String s : lineArr) {
						if("0".equals(s)) {
							j++;
							continue;
						}
						testData.put(0, j++, Integer.valueOf(s));
					}
				}
				double mindis = 0.0;
				int label = 0;
				double diszero = Transforms.allEuclideanDistances(testData, this.zero, 1).sumNumber().doubleValue() / this.zero.shape()[0];
				mindis = diszero;
				double disone = Transforms.allEuclideanDistances(testData, this.one, 1).sumNumber().doubleValue() / this.one.shape()[0];
				if(disone < mindis) {
					mindis = disone;
					label = 1;
				}
				double distwo = Transforms.allEuclideanDistances(testData, this.two, 1).sumNumber().doubleValue() / this.two.shape()[0];
				if(distwo < mindis) {
					mindis = distwo;
					label = 2;
				}
				double disthree = Transforms.allEuclideanDistances(testData, this.three, 1).sumNumber().doubleValue() / this.three.shape()[0];
				if(disthree < mindis) {
					mindis = disthree;
					label = 3;
				}
				double disfour = Transforms.allEuclideanDistances(testData, this.four, 1).sumNumber().doubleValue() / this.four.shape()[0];
				if(disfour < mindis) {
					mindis = disfour;
					label = 4;
				}
				double disfive = Transforms.allEuclideanDistances(testData, this.five, 1).sumNumber().doubleValue() / this.five.shape()[0];
				if(disfive < mindis) {
					mindis = disfive;
					label = 5;
				}
				double dissix = Transforms.allEuclideanDistances(testData, this.six, 1).sumNumber().doubleValue() / this.six.shape()[0];
				if(dissix < mindis) {
					mindis = dissix;
					label = 6;
				}
				double disseven = Transforms.allEuclideanDistances(testData, this.seven, 1).sumNumber().doubleValue() / this.seven.shape()[0];
				if(disseven < mindis) {
					mindis = disseven;
					label = 7;
				}
				double diseight = Transforms.allEuclideanDistances(testData, this.eight, 1).sumNumber().doubleValue() / this.eight.shape()[0];
				if(diseight < mindis) {
					mindis = diseight;
					label = 8;
				}
				double disnine = Transforms.allEuclideanDistances(testData, this.nine, 1).sumNumber().doubleValue() / this.nine.shape()[0];
				if(disnine < mindis) {
					mindis = disnine;
					label = 9;
				}
				if(label != curLabel) {
					errorCount++;
					System.out.println("正确数字：" + curLabel + "，识别为：" + label + "，文件名：" + testFile.getName());
				}
			} catch(Exception e) {
				
			}
		}
		System.out.println("错误数：" + errorCount);
		System.out.println("错误率：" + (errorCount * 100 / testFiles.length) + "%");
	}
}
