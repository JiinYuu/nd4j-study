package com.lmq.ml.nd4j.knn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.util.StringUtils;

public class DatingDecisionMaker {

	private INDArray data, label;
	
	private Map<Integer, INDArray> dataMap;
	private Map<Integer, INDArray> testMap;
	
	private INDArray testd, testl;
	
	private static final String DATING_DATA_FILE = "knn/datingTestSet2.txt";
	
	public void init() {
		File file = new File(DatingDecisionMaker.class.getClassLoader().getResource(DATING_DATA_FILE).getFile());
		Map<Integer, List<double[]>> dataMap = new HashMap<>(3);
		try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			List<double[]> dataList = reader.lines().filter(StringUtils::hasText).map(line -> {
				String[] lineArr = line.split("\t");
				double[] d = new double[lineArr.length];
				Arrays.setAll(d, i -> Double.parseDouble(lineArr[i]));
				
				double[] d2 = Arrays.copyOf(d, d.length - 1);
				int label = (int) d[d.length - 1];
				
				if(dataMap.containsKey(label)) {
					dataMap.get(label).add(d2);
				} else {
					List<double[]> d2s = new ArrayList<>();
					d2s.add(d2);
					dataMap.put(label, d2s);
				}
				
				return d;
			}).collect(Collectors.toList());
			
			Random random = new Random();
			Set<Integer> randIndex = new HashSet<>(dataList.size() * 3 / 10);
			List<double[]> testDataList = new ArrayList<>(dataList.size() * 3 / 10);
			for(int i = 0; i < dataList.size() * 3 / 10; i++) {
				int rand = random.nextInt(dataList.size());
				if(randIndex.add(rand)) {
					testDataList.add(dataList.get(rand));
				}
			}
			
			dataList.removeAll(testDataList);
			
			double[][] datas = dataList.toArray(new double[dataList.size()][4]);
			this.data = Nd4j.create(datas);
			this.label = this.data.getColumn(3);
			this.data = this.data.getColumns(0, 1, 2);
			
			double[][] testDatas = testDataList.toArray(new double[testDataList.size()][4]);
			this.testd = Nd4j.create(testDatas);
			this.testl = this.testd.getColumn(3);
			this.testd = this.testd.getColumns(0, 1, 2);
			
			INDArray min = this.data.min(0);
			INDArray max = this.data.max(0);
			INDArray mint = this.testd.min(0);
			INDArray maxt = this.testd.max(0);
			min = Transforms.min(min, mint);
			max = Transforms.max(max, maxt);
			this.data = this.data.subRowVector(min).divRowVector(max.sub(min));
			this.testd = this.testd.subRowVector(min).divRowVector(max.sub(min));
			
			this.dataMap = new HashMap<>(3);
			this.testMap = new HashMap<>(3);
			for(Map.Entry<Integer, List<double[]>> entry : dataMap.entrySet()) {
				List<double[]> dsList = entry.getValue();
				Set<Integer> randIndexLocal = new HashSet<>(dsList.size() * 3 / 10);
				List<double[]> testDataListLocal = new ArrayList<>(dsList.size() * 3 / 10);
				for(int i = 0; i < dsList.size() * 3 / 10; i++) {
					int rand = random.nextInt(dsList.size());
					if(randIndexLocal.add(rand)) {
						testDataListLocal.add(dsList.get(rand));
					}
				}
				dsList.removeAll(testDataListLocal);
				double[][] ds = dsList.toArray(new double[dsList.size()][3]);
				INDArray nd = Nd4j.create(ds);
				nd = nd.subRowVector(min).divRowVector(max.sub(min));
				this.dataMap.put(entry.getKey(), nd);
				
				double[][] testDs = testDataListLocal.toArray(new double[testDataListLocal.size()][3]);
				INDArray testNd = Nd4j.create(testDs);
				testNd = testNd.subRowVector(min).divRowVector(max.sub(min));
				this.testMap.put(entry.getKey(), testNd);
			}
			
		} catch(Exception e) {
			
		}
	}
	
	public double test(int k) {
		int m = this.testd.shape()[0];
		int error = 0;
		for(int i = 0; i < m; i++) {
			INDArray test = this.testd.getRow(i);
			int curLabel = this.testl.getInt(i, 0);
			int length = this.data.shape()[0];
			List<Pair<Integer, Double>> distances = new ArrayList<>(length);
			INDArray dis = Transforms.allEuclideanDistances(test, this.data, 1);
			for(int j = 0; j < length; j++) {
				Pair<Integer, Double> disPair = new Pair<>(this.label.getInt(j, 0), dis.getDouble(0, j));
				distances.add(disPair);
			}
			Collections.sort(distances, (p1, p2) -> p1.getValue().compareTo(p2.getValue()));
			int label = distances.stream().sorted((p1, p2) -> p1.getValue().compareTo(p2.getValue())).limit(k).collect(Collectors.groupingBy(Pair::getKey)).entrySet().stream().sorted((e1, e2) -> e2.getValue().size() - e1.getValue().size()).findFirst().get().getKey();
			if(curLabel != label) {
				error++;
			}
		}
		System.out.println("错误数：" + error);
		System.out.println("错误率：" + (error * 100.0) / m);
		return error * 100.0 / m;
	}
	
	public double test2(int k) {
		
		int error = 0, sum = 0;
		for(Entry<Integer, INDArray> entry : this.testMap.entrySet()) {
			int curLabel = entry.getKey();
			INDArray testd = entry.getValue();
			int m = entry.getValue().shape()[0];
			sum += m;
			
			for(int i = 0; i < m; i++) {
				INDArray test = testd.getRow(i);
				List<Pair<Integer, Double>> distances = new ArrayList<>(this.dataMap.size());
				for(Entry<Integer, INDArray> entry2 : this.dataMap.entrySet()) {
					INDArray dataLocal = entry2.getValue();
//					int length = dataLocal.shape()[0];
					int[] columns = new int[4];
					Arrays.setAll(columns, num -> num);
					double[] diss = Transforms.allEuclideanDistances(test, dataLocal, 1).toDoubleVector();
					final AtomicInteger index = new AtomicInteger(0);
					double dis = Stream.generate(() -> diss[index.getAndIncrement()]).limit(diss.length).sorted().limit(k).reduce((d1, d2) -> d1 + d2).get() / k;
					
//					double dis = Transforms.allEuclideanDistances(test, dataLocal, 1).getColumns(columns).sumNumber().doubleValue() / length;
					distances.add(new Pair<>(entry2.getKey(), dis));
				}
				int label = distances.stream().sorted((p1, p2) -> p1.getValue().compareTo(p2.getValue())).findFirst().get().getKey();
				if(curLabel != label) {
					error++;
				}
			}
		}
		
		
		System.out.println("错误数：" + error);
		System.out.println("错误率：" + (error * 100.0) / sum);
		return error * 100.0 / sum;
	}
}
