package com.lmq.ml.nd4j.trees;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.Pair;
import org.springframework.util.StringUtils;

public class LensSelector {

	private static final String TRAINING_DATA_FILE = "trees/lenses.txt";

	public List<Map<String, String>> data;
	public String[] labels = { "age", "prescript", "astigmatic", "tearRate" };
	
	public void init() {
		File trainingFile = new File(LensSelector.class.getClassLoader().getResource(TRAINING_DATA_FILE).getFile());
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainingFile)))) {
			data = new ArrayList<>();
			String line = null;
			while(StringUtils.hasText((line = reader.readLine()))) {
				String[] lineArr = line.split("\t");
				Map<String, String> sample = new HashMap<>(lineArr.length);
				for(int i = 0; i < labels.length; i++) {
					sample.put(labels[i], lineArr[i]);
				}
				sample.put("class", lineArr[lineArr.length - 1]);
				data.add(sample);
			}
			
		} catch (Exception e) {
			
		}
	}
	
	public Object createTree(List<Map<String, String>> data, List<String> labels) {
		
		Map<String, Pair<String, Integer>> classCount = new HashMap<>();
		for(Map<String, String> sample : data) {
			String clazz = sample.get("class");
			if(classCount.containsKey(clazz)) {
				classCount.put(clazz, new Pair<>(clazz, classCount.get(clazz).getValue() + 1));
			} else {
				classCount.put(clazz, new Pair<>(clazz, 1));
			}
		}
		if(classCount.keySet().size() == 1) {
			return classCount.keySet().iterator().next();
		}
		if(data.get(0).size() == 1) {
			return classCount.values().stream().sorted((p1, p2) -> p2.getValue().compareTo(p1.getValue())).collect(Collectors.toList()).get(0).getKey();
		}
		String best = this.chooseBestFeature(data, labels);
		Map<String, Map<String, Object>> tree = new LinkedHashMap<>(1);
		Set<String> labelValues = data.stream().map(m -> m.get(best)).collect(Collectors.toSet());
		tree.put(best, new LinkedHashMap<String, Object>(labelValues.size()));
		List<String> subLabels = new ArrayList<>(labels.size() - 1);
		subLabels.addAll(labels);
		subLabels.remove(best);
		for(String value : labelValues) {
			List<Map<String, String>> subData = data.stream().filter(m -> value.equals(m.get(best))).map(m -> {
				Map<String, String> ret = new HashMap<>(m.size() - 1);
				ret.putAll(m);
				ret.remove(best);
				return ret;
			}).collect(Collectors.toList());
			
			tree.get(best).put(value, createTree(subData, subLabels));
		}
		return tree;
	}
	
	public String chooseBestFeature(List<Map<String, String>> data, List<String> labels) {
		double shannon = this.calcShannonEnt(data);
		double infoGain = 0.0;
		String feature = null;
		for(String label : labels) {
			Set<String> labelValues = data.stream().map(m -> m.get(label)).collect(Collectors.toSet());
			double shannon2 = 0.0;
			for(String labelValue : labelValues) {
				List<Map<String, String>> subData = data.stream().filter(m -> labelValue.equals(m.get(label))).collect(Collectors.toList());
				shannon2 += (subData.size() / Double.valueOf(data.size())) * calcShannonEnt(subData);
			}
			if(shannon - shannon2 > infoGain) {
				infoGain = shannon - shannon2;
				feature = label;
			}
		}
		return feature;
	}
	
	public double calcShannonEnt(List<Map<String, String>> data) {
		Map<String, Pair<String, Integer>> classCount = new HashMap<>();
		for(Map<String, String> sample : data) {
			String clazz = sample.get("class");
			if(classCount.containsKey(clazz)) {
				classCount.put(clazz, new Pair<>(clazz, classCount.get(clazz).getValue() + 1));
			} else {
				classCount.put(clazz, new Pair<>(clazz, 1));
			}
		}
		double ret = 0.0;
		for(Map.Entry<String, Pair<String, Integer>> entry : classCount.entrySet()) {
			double prop = entry.getValue().getValue() / Double.valueOf(data.size());
			ret -= prop * (Math.log(prop) / Math.log(2));
		}
		return ret;
	}
	
	@SuppressWarnings("unchecked")
	public String classify(Object tree, Map<String, String> sample) {
		if(tree instanceof String) {
			return tree.toString();
		}
		String first = ((Map<String, Object>) tree).keySet().iterator().next();
		Map<String, Object> map = (Map<String, Object>) ((Map<String, Object>) tree).get(first);
		Object obj = map.get(sample.get(first));
		if(obj instanceof Map) {
			return classify((Map<String, Object>) obj, sample);
		} else {
			return obj.toString();
		}
	}
}
