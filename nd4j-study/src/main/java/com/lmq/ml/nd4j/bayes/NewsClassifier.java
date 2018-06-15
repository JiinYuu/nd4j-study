package com.lmq.ml.nd4j.bayes;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.wltea.analyzer.IKSegmentation;
import org.wltea.analyzer.Lexeme;

import com.lmq.ml.nd4j.bayes.NewsProvider.News;

public class NewsClassifier {

	private INDArray title;
	private INDArray content;
	private INDArray label;
	
	private INDArray pst, psc, pft, pfc;
	
	private INDArray testTitle;
	private INDArray testContent;
	private INDArray testLabel;
	
	private NewsProvider np = new NewsProvider();
	
	public void init() throws IOException {
		List<News> sports = np.sports();
		List<News> finance = np.finance();
		Random random = new Random();
		Set<Integer> testSportsIndex = new HashSet<>(30);
		Set<Integer> testFinanceIndex = new HashSet<>(30);
		List<News> testSports = new ArrayList<>(30);
		List<News> testFinance = new ArrayList<>(30);
		while(true) {
			int rand = random.nextInt(sports.size());
			if(testSportsIndex.size() < 30) {
				if(testSportsIndex.add(rand)) {
					testSports.add(sports.get(rand));
				}
			}
			int rand2 = random.nextInt(finance.size());
			if(testFinanceIndex.size() < 30) {
				if(testFinanceIndex.add(rand2)) {
					testFinance.add(finance.get(rand2));
				}
			}
			if(testSportsIndex.size() >= 30 && testFinanceIndex.size() >= 30) {
				break;
			}
		}
		sports.removeAll(testSports);
		finance.removeAll(testFinance);
		
		Set<String> titleWords = new HashSet<>();
		Set<String> contentWords = new HashSet<>();
		List<Pair<Integer, Pair<List<String>, List<String>>>> labelsAndWords = new ArrayList<>();
		for(int i = 0; i < sports.size(); i++) {
			News news = sports.get(i);
			List<String> tws = this.cutWords(news.getTitle());
			List<String> cws = this.cutWords(news.getContent());
			titleWords.addAll(tws);
			contentWords.addAll(cws);
			Pair<List<String>, List<String>> words = new Pair<>(tws, cws);
			Pair<Integer, Pair<List<String>, List<String>>> law = new Pair<>((int) 's', words);
			labelsAndWords.add(law);
		}
		for(int i = 0; i < finance.size(); i++) {
			News news = finance.get(i);
			List<String> tws = this.cutWords(news.getTitle());
			List<String> cws = this.cutWords(news.getContent());
			titleWords.addAll(tws);
			contentWords.addAll(cws);
			Pair<List<String>, List<String>> words = new Pair<>(tws, cws);
			Pair<Integer, Pair<List<String>, List<String>>> law = new Pair<>((int) 'f', words);
			labelsAndWords.add(law);
		} 
		
		String[] titleWordsArr = titleWords.toArray(new String[titleWords.size()]);
		String[] contentWordsArr = contentWords.toArray(new String[contentWords.size()]);
		this.label = Nd4j.zeros(sports.size() + finance.size(), 1);
		this.title = Nd4j.zeros(sports.size() + finance.size(), titleWordsArr.length);
		this.content = Nd4j.zeros(sports.size() + finance.size(), contentWordsArr.length);
		for(int i = 0; i < labelsAndWords.size() - finance.size(); i++) {
			Pair<Integer, Pair<List<String>, List<String>>> law = labelsAndWords.get(i);
			this.label.put(i, 0, law.getKey());
			
			Map<String, List<String>> tws = law.getValue().getFirst().stream().collect(Collectors.groupingBy(String::toString));
			for(int j = 0; j < titleWordsArr.length; j++) {
				if(tws.containsKey(titleWordsArr[j])) {
					this.title.put(i, j, 1 + tws.get(titleWordsArr[j]).size());
				}
			}
			Map<String, List<String>> cws = law.getValue().getSecond().stream().collect(Collectors.groupingBy(String::toString));
			for(int j = 0; j < contentWordsArr.length; j++) {
				if(cws.containsKey(contentWordsArr[j])) {
					this.content.put(i, j, 1 + cws.get(contentWordsArr[j]).size());
				}
			}
		}
		INDArray pstsum = Nd4j.ones(1, titleWordsArr.length).add(this.title.sum(0));
		INDArray pstsumAll = pstsum.sum(1);
		this.pst = Transforms.log(pstsum.div(pstsumAll), 2);
		INDArray pscsum = Nd4j.ones(1, contentWordsArr.length).add(this.content.sum(0));
		INDArray pscsumAll = pscsum.sum(1);
		this.psc = Transforms.log(pscsum.div(pscsumAll), 2);
		
		for(int i = sports.size(); i < labelsAndWords.size(); i++) {
			Pair<Integer, Pair<List<String>, List<String>>> law = labelsAndWords.get(i);
			this.label.put(i, 0, law.getKey());
			
			Map<String, List<String>> tws = law.getValue().getFirst().stream().collect(Collectors.groupingBy(String::toString));
			for(int j = 0; j < titleWordsArr.length; j++) {
				if(tws.containsKey(titleWordsArr[j])) {
					this.title.put(i, j, 1 + tws.get(titleWordsArr[j]).size());
				}
			}
			Map<String, List<String>> cws = law.getValue().getSecond().stream().collect(Collectors.groupingBy(String::toString));
			for(int j = 0; j < contentWordsArr.length; j++) {
				if(cws.containsKey(contentWordsArr[j])) {
					this.content.put(i, j, 1 + cws.get(contentWordsArr[j]).size());
				}
			}
		}
		INDArray pftsum = Nd4j.ones(1, titleWordsArr.length).add(this.title.sum(0).sub(pstsum.sub(1)));
		INDArray pftsumAll = pftsum.sum(1);
		this.pft = Transforms.log(pftsum.div(pftsumAll), 2);
		INDArray pfcsum = Nd4j.ones(1, contentWordsArr.length).add(this.content.sum(0).sub(pscsum.sub(1)));
		INDArray pfcsumAll = pfcsum.sum(1);
		this.pfc = Transforms.log(pfcsum.div(pfcsumAll), 2);
		
		List<Pair<Integer, Pair<List<String>, List<String>>>> testLabelsAndWords = new ArrayList<>();
		for(int i = 0; i < testSports.size(); i++) {
			News news = testSports.get(i);
			List<String> tws = this.cutWords(news.getTitle());
			List<String> cws = this.cutWords(news.getContent());
			Pair<List<String>, List<String>> words = new Pair<>(tws, cws);
			Pair<Integer, Pair<List<String>, List<String>>> law = new Pair<>((int) 's', words);
			testLabelsAndWords.add(law);
		}
		for(int i = 0; i < testFinance.size(); i++) {
			News news = testFinance.get(i);
			List<String> tws = this.cutWords(news.getTitle());
			List<String> cws = this.cutWords(news.getContent());
			Pair<List<String>, List<String>> words = new Pair<>(tws, cws);
			Pair<Integer, Pair<List<String>, List<String>>> law = new Pair<>((int) 'f', words);
			testLabelsAndWords.add(law);
		} 
		this.testLabel = Nd4j.zeros(testSports.size() + testFinance.size(), 1);
		this.testTitle = Nd4j.zeros(testSports.size() + testFinance.size(), titleWordsArr.length);
		this.testContent = Nd4j.zeros(testSports.size() + testFinance.size(), contentWordsArr.length);
		for(int i = 0; i < testLabelsAndWords.size(); i++) {
			Pair<Integer, Pair<List<String>, List<String>>> law = testLabelsAndWords.get(i);
			this.testLabel.put(i, 0, law.getKey());
			
			Map<String, List<String>> tws = law.getValue().getFirst().stream().collect(Collectors.groupingBy(String::toString));
			for(int j = 0; j < titleWordsArr.length; j++) {
				if(tws.containsKey(titleWordsArr[j])) {
					this.testTitle.put(i, j, 1 + tws.get(titleWordsArr[j]).size());
				}
			}
			Map<String, List<String>> cws = law.getValue().getSecond().stream().collect(Collectors.groupingBy(String::toString));
			for(int j = 0; j < contentWordsArr.length; j++) {
				if(cws.containsKey(contentWordsArr[j])) {
					this.testContent.put(i, j, 1 + cws.get(contentWordsArr[j]).size());
				}
			}
		}
	}
	
	public Pair<Double, Double> test() {
		int errorCount = 0;
		int index = 0;
		int errorCount2 = 0, index2 = 0;
		INDArray psts = Nd4j.zeros(testLabel.shape());
		INDArray pscs = Nd4j.zeros(testLabel.shape());
		INDArray pfts = Nd4j.zeros(testLabel.shape());
		INDArray pfcs = Nd4j.zeros(testLabel.shape());
		for(int i = 0; i < this.testLabel.shape()[0]; i++) {
			INDArray content = this.testContent.getRow(i);
			INDArray title = this.testTitle.getRow(i);
			double pst = title.mul(this.pst).sum(1).getDouble(0, 0);
			double psc = content.mul(this.psc).sum(1).getDouble(0, 0);
			psts.put(i, 0, pst);
			pscs.put(i, 0, psc);
			double pft = title.mul(this.pft).sum(1).getDouble(0, 0);
			double pfc = content.mul(this.pfc).sum(1).getDouble(0, 0);
			pfts.put(i, 0, pft);
			pfcs.put(i, 0, pfc);
			
			double ps = pst * 0.5 + psc * 0.5;
			double pf = pft * 0.5 + pfc * 0.5;
			int curLabel = testLabel.getInt(i, 0);
			int label = ps > pf ? 's' : 'f';
			if(curLabel != label) {
				errorCount2++;
			}
			index2++;
		}
		System.out.println("----错误数：" + errorCount2 + "，错误率：" + (errorCount2 / Double.valueOf(index2)));
		
		double pstmin = psts.minNumber().doubleValue();
		double pstmax = psts.maxNumber().doubleValue();
		psts = psts.sub(pstmin).div(pstmax - pstmin);
		
		double pscmin = pscs.minNumber().doubleValue();
		double pscmax = pscs.maxNumber().doubleValue();
		pscs = pscs.sub(pscmin).div(pscmax - pscmin);
		
		double pftmin = pfts.minNumber().doubleValue();
		double pftmax = pfts.maxNumber().doubleValue();
		pfts = pfts.sub(pftmin).div(pftmax - pftmin);
		
		double pfcmin = pfcs.minNumber().doubleValue();
		double pfcmax = pfcs.maxNumber().doubleValue();
		pfcs = pfcs.sub(pfcmin).div(pfcmax - pfcmin);
		
		for(int i = 0; i < testLabel.shape()[0]; i++) {
			int curLabel = testLabel.getInt(i, 0);
			double pst = psts.getDouble(i, 0);
			double psc = pscs.getDouble(i, 0);
			double ps = pst * 0.0 + psc * 0.1;
			double pft = pfts.getDouble(i, 0);
			double pfc = pfcs.getDouble(i, 0);
			double pf = pft * 0.0 + pfc * 0.1;
			int label = ps > pf ? 's' : 'f';
			if(curLabel != label) {
				errorCount++;
			}
			index++;
		}
		System.out.println("归一化错误数：" + errorCount + "，错误率：" + (errorCount / Double.valueOf(index)));
		return new Pair<>((errorCount2 / Double.valueOf(index2)), (errorCount / Double.valueOf(index)));
	}
	
	private List<String> cutWords(String str) throws IOException {
		str = str.replaceAll("<[^>]*>", "");
		IKSegmentation ik = new IKSegmentation(new StringReader(str), true);
		Lexeme word = ik.next();
		List<String> ret = new ArrayList<>();
		while(word != null) {
			ret.add(word.getLexemeText());
			word = ik.next();
		}
		return ret;
	}
}
