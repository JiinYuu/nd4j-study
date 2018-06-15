package com.lmq.ml.nd4j.logistic;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;
import org.nd4j.linalg.ops.transforms.Transforms;

public class HorseColic {

	private INDArray data, label;
	private static final String TRAINING_DATA_FILE = "logistic/horseColicTraining.txt";
	private static final String TEST_DATA_FILE = "logistic/horseColicTest.txt";
	
	public void init() {
		File file = new File(HorseColic.class.getClassLoader().getResource(TRAINING_DATA_FILE).getFile());
		try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			List<double[]> datas = reader.lines().filter(line -> StringUtils.hasText(line)).map(line -> {
				String[] ss = line.split("\t");
				double[] ds = new double[ss.length + 1];
				Arrays.setAll(ds, i -> {
					if(i < ss.length - 1) {
						return Double.parseDouble(ss[i]);
					}
					if(i == ss.length - 1) {
						return 1.0;
					}
					return Double.parseDouble(ss[ss.length - 1]);
				});
				return ds;
			}).collect(Collectors.toList());
			int m = datas.size();
			int n = datas.get(0).length - 1;
			this.data = Nd4j.create(datas.toArray(new double[m][n + 1]));
			this.label = this.data.getColumn(n);
			int[] columns = new int[n];
			Arrays.setAll(columns, i -> i);
			this.data = this.data.getColumns(columns);
//			System.out.println(this.data.toString());
//			System.out.println(this.label.toString());
		} catch(Exception e) {
			
		}
	}
	
	public INDArray logisticRegres() {
		INDArray weights = Nd4j.ones(this.data.shape()[1], 1);
		float alpha = 0.001F;
		int maxCycles = 500;
		for(int i = 0; i < maxCycles; i++) {
			INDArray h = Transforms.sigmoid(this.data.mmul(weights));
			INDArray error = this.label.sub(h);
			weights = weights.add(this.data.transpose().mmul(error).mul(alpha));
		}
		return weights;
	}
	
	public INDArray logisticRegres2() {
		INDArray weights = Nd4j.ones(this.data.shape()[1], 1);
		double alpha = 0.001;
		int maxCycles = 30;
		Random random = new Random();
		for(int i = 0; i < maxCycles; i++) {
			for(int j = 0; j < this.data.shape()[0]; j++) {
				alpha = 4 / (1.0 + i + j) + 0.0001;
				int randIndex = random.nextInt(this.data.shape()[0]);
				INDArray row = this.data.getRow(randIndex);
				INDArray h = Transforms.sigmoid(row.mmul(weights));
				INDArray error = this.label.getRow(randIndex).sub(h);
				weights = weights.add(row.transpose().mmul(error.mul(alpha)));
			}
		}
		return weights;
	}
	
	public void test() {
		this.init();
		INDArray weights = this.logisticRegres();
		INDArray testd, testl;
		File file = new File(HorseColic.class.getClassLoader().getResource(TEST_DATA_FILE).getFile());
		try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			List<double[]> datas = reader.lines().filter(line -> StringUtils.hasText(line)).map(line -> {
				String[] ss = line.split("\t");
				double[] ds = new double[ss.length + 1];
				Arrays.setAll(ds, i -> {
					if(i < ss.length - 1) {
						return Double.parseDouble(ss[i]);
					}
					if(i == ss.length - 1) {
						return 1.0;
					}
					return Double.parseDouble(ss[ss.length - 1]);
				});
				return ds;
			}).collect(Collectors.toList());
			int m = datas.size();
			int n = datas.get(0).length - 1;
			testd = Nd4j.create(datas.toArray(new double[m][n + 1]));
			testl = testd.getColumn(n);
			int[] columns = new int[n];
			Arrays.setAll(columns, i -> i);
			testd = testd.getColumns(columns);
			
			int errorCount = 0;
			for(int i = 0; i < m; i++) {
				int curLabel = testl.getInt(i, 0);
				INDArray row = testd.getRow(i);
				double prop = Transforms.sigmoid(row.mmul(weights)).getDouble(0, 0);
				int label = prop > 0.5 ? 1 : 0;
				if(curLabel != label) {
					errorCount++;
				}
			}
			System.out.println("错误数：" + errorCount);
			System.out.println("错误率：" + (errorCount * 100 / m) + "%");
			
		} catch(Exception e) {
			
		}
	}
	
	public void test2() {
		this.init();
		INDArray weights = this.logisticRegres2();
		INDArray testd, testl;
		File file = new File(HorseColic.class.getClassLoader().getResource(TEST_DATA_FILE).getFile());
		try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			List<double[]> datas = reader.lines().filter(line -> StringUtils.hasText(line)).map(line -> {
				String[] ss = line.split("\t");
				double[] ds = new double[ss.length + 1];
				Arrays.setAll(ds, i -> {
					if(i < ss.length - 1) {
						return Double.parseDouble(ss[i]);
					}
					if(i == ss.length - 1) {
						return 1.0;
					}
					return Double.parseDouble(ss[ss.length - 1]);
				});
				return ds;
			}).collect(Collectors.toList());
			int m = datas.size();
			int n = datas.get(0).length - 1;
			testd = Nd4j.create(datas.toArray(new double[m][n + 1]));
			testl = testd.getColumn(n);
			int[] columns = new int[n];
			Arrays.setAll(columns, i -> i);
			testd = testd.getColumns(columns);
			
			int errorCount = 0;
			for(int i = 0; i < m; i++) {
				int curLabel = testl.getInt(i, 0);
				INDArray row = testd.getRow(i);
				double prop = Transforms.sigmoid(row.mmul(weights)).getDouble(0, 0);
				int label = prop > 0.5 ? 1 : 0;
				if(curLabel != label) {
					errorCount++;
				}
			}
			System.out.println("错误数：" + errorCount);
			System.out.println("错误率：" + (errorCount * 100 / m) + "%");
			
		} catch(Exception e) {
			
		}
	}
	
	public void testClassify() {
		this.init();
		INDArray weights = this.logisticRegres();
		int m = this.data.shape()[0];
		int errorCount = 0;
		for(int i = 0; i < m; i++) {
			int curLabel = this.label.getInt(i, 0);
			INDArray row = this.data.getRow(i);
			double prop = Transforms.sigmoid(row.mmul(weights)).getDouble(0, 0);
			int label = prop > 0.5 ? 1 : 0;
			if(curLabel != label) {
				errorCount++;
			}
		}
		System.out.println("错误数：" + errorCount);
		System.out.println("错误率：" + (errorCount * 100 / m) + "%");
	}
	
	public void testClassify2() {
		this.init();
		INDArray weights = this.logisticRegres2();
		int m = this.data.shape()[0];
		int errorCount = 0;
		for(int i = 0; i < m; i++) {
			int curLabel = this.label.getInt(i, 0);
			INDArray row = this.data.getRow(i);
			double prop = Transforms.sigmoid(row.mmul(weights)).getDouble(0, 0);
			int label = prop > 0.5 ? 1 : 0;
			if(curLabel != label) {
				errorCount++;
			}
		}
		System.out.println("错误数：" + errorCount);
		System.out.println("错误率：" + (errorCount * 100 / m) + "%");
	}
}
