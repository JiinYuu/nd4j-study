package com.lmq.ml.nd4j.logistic;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SimpleSample {

	private INDArray data;
	
	private INDArray label;
	
	private static final String TRAINING_DATA_FILE = "logistic/testSet.txt";
	
	public void init() {
		File file = new File(SimpleSample.class.getClassLoader().getResource(TRAINING_DATA_FILE).getFile());
		try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			List<float[]> datass = reader.lines().map(line -> {
				String[] lineArr = line.split("\t");
				float[] datas = new float[lineArr.length + 1];
				datas[0] = Float.valueOf(lineArr[0]);
				datas[1] = Float.valueOf(lineArr[1]);
				datas[2] = 1.0F;
				datas[3] = Integer.valueOf(lineArr[2]);
				return datas;
			}).collect(Collectors.toList());
			this.data = Nd4j.create(datass.toArray(new float[datass.size()][4]));
			this.label = this.data.getColumn(3);
			this.data = this.data.getColumns(0, 1, 2);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public INDArray logisticRegres() {
		INDArray weights = Nd4j.ones(3, 1);
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
		INDArray weights = Nd4j.ones(3, 1);
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
