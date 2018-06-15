package com.lmq.ml.ndfj.bayes;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.wltea.analyzer.IKSegmentation;
import org.wltea.analyzer.Lexeme;

public class IKTest {

	private IKSegmentation ik = new IKSegmentation(new StringReader("<p style=\"text-indent: 2em;\"> 人民网北京5月25日电（胡雪蓉） 5月25日，北京冬奥组委发布征集公告，面向国内征集首批北京2022年冬奥会和冬残奥会（以下简称北京冬奥会）特许生产商。此次公开征集徽章、钥匙扣及其他非贵金属制品，服装服饰及配饰，丝绸制品，贵金属制品，文具，陶瓷制品（含紫砂壶）六大产品类别的特许生产商，负责相应特许商品的开发、设计和组织生产。除贵金属类别计划征集5家特许生产商外，其他类别各计划征集2-3家特许生产商。</p> <p style=\"text-indent: 2em;\"> 北京冬奥组委对应征特许生产商的企业提出了严格要求，主要包括：企业资信状况及社会信誉良好；企业注册成立2年以上，具备一般纳税人资质；注册资本在人民币2000万元以上；通过质量管理体系认证、环境管理体系认证，或省级以上相关部门质量、环保检测达标；通过信息安全管理体系认证或已建立安全有效的信息管理体系；具备良好的产品开发、设计、组织生产、市场营销及抗风险能力。</p> <p style=\"text-indent: 2em;\"> 本次公开征集将按以下程序进行：北京冬奥组委在官方网站发布征集公告；应征企业在规定时间内向北京冬奥组委提交《应征人意向函》及相关资质证明；北京冬奥组委向符合条件的应征企业发放征集书，应征企业在规定时间内提交应征文件；北京冬奥组委特许经营企业征集评审组对应征文件进行评审，并将组成考察组对特许生产商候选企业进行实地考察。最终确定的特许生产商中选企业将与北京冬奥组委签订特许经营协议。</p> <p style=\"text-indent: 2em;\"> 有意向参与征集的企业须于5月30日前向北京冬奥组委提交意向函，并附上企业的营业执照及质量管理体系认证、环保管理体系认证等资质证明材料。通过邮件方式发送至特许经营企业征集邮箱（licensing@beijing2022.cn），也可当面递交到北京冬奥组委市场开发部。</p> <p style=\"text-indent: 2em;\"> 北京冬奥会特许经营计划是北京冬奥会市场开发计划的一部分，其主要目的是弘扬奥林匹克精神，传播北京冬奥会理念，提升冬奥会和冬残奥会品牌价值，宣传中国优秀传统文化和主办城市特色，为国内广大中小企业和社会公众参与和支持奥运搭建平台。2017年12月15日，北京冬奥组委发布北京冬奥会会徽和冬残奥会会徽后，为扩大会徽宣传，满足广大消费者和奥运收藏爱好者对会徽商品的购买需求，北京冬奥组委启动了特许经营试运行计划，开发了徽章、钥匙扣等非贵金属制品，贵金属制品，服装服饰，文具，陶瓷及邮票品等六大类、200余款特许商品上市销售，在北京、石家庄、上海、南京、张家口等地开设了16个特许商品零售店，并上线运行了北京2022特许商品官方网店，得到广大消费者的欢迎。根据北京冬奥会市场开发计划相关安排，将于2018年下半年正式启动北京2022年冬奥会和冬残奥会特许经营计划。</p>".replaceAll("<[^>]*>", "")), true);
	
	public @Test void testIk() throws IOException {
		List<String> words = new ArrayList<>();
		Lexeme word = ik.next();
		while(word != null) {
			words.add(word.getLexemeText());
			word = ik.next();
		}
		System.out.println(words);
	}
	
	public @Test void testReplaceHtml() {
		String str = "<p style=\"text-indent: 2em;\"> 人民网北京5月25日电（胡雪蓉） 5月25日，</p>";
		System.out.println(str.replaceAll("<[^>]*>", ""));
	}
}
