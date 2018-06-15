package com.lmq.ml.nd4j.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.springframework.web.client.RestTemplate;

import lombok.Data;

public class NewsProvider {

	private static final String NEWS_URL = "http://localhost:5000/news/";
	private RestTemplate rest = new RestTemplate();
	
	public List<News> sports() {
		News[] newes = rest.getForObject(NEWS_URL + "sports", News[].class);
		return new ArrayList<>(Arrays.asList(newes));
	}
	
	public List<News> finance() {
		News[] newes = rest.getForObject(NEWS_URL + "finance", News[].class);
		return new ArrayList<>(Arrays.asList(newes));
	}
	
	
	public @Data static class News {
		private String title, content;
	}
}
