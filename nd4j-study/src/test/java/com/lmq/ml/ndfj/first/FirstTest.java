package com.lmq.ml.ndfj.first;

import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lmq.ml.nd4j.test.NDMatchers;

public class FirstTest {
	
	public @Test void testCreate() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 }); // 2 * 2
		Assert.assertThat(nd.shape(), Matchers.is(new int[] { 2, 2 }));
		Assert.assertThat(nd.getFloat(0, 1), Matchers.is(2F));
		Assert.assertArrayEquals(new float[] { 2, 4 }, nd.getColumn(1).toFloatVector(), 0F);
		nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 }); // 3 * 2
		Assert.assertThat(nd.getRow(2).toFloatVector(), Matchers.is(new float[] { 5, 6 }));
		Assert.assertThat(nd.getColumn(1).toFloatVector(), Matchers.is(new float[] { 2, 4, 6 }));
		nd = Nd4j.create(new float[][] {{ 1, 3, 5, 7}, { 2, 4, 6, 8}}); // 一行行叠加，所以2 * 4
		Assert.assertArrayEquals(new int[] { 2, 4}, nd.shape());
		Assert.assertArrayEquals(new float [] { 5, 6 }, nd.getColumn(2).toFloatVector(), 0F);
	}
	
	public @Test void testScalarOperations() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new int[] { 4, 2 }); // 4 * 2
		Assert.assertArrayEquals(new float[] { 5, 6 }, nd.getRow(2).toFloatVector(), 0F);
		INDArray nd2 = nd.add(1);
		Assert.assertArrayEquals(new float[] { 5, 6 }, nd.getRow(2).toFloatVector(), 0F);
		Assert.assertArrayEquals(new float[] { 4, 5 }, nd2.getRow(1).toFloatVector(), 0F);
		nd.addi(1);
		Assert.assertArrayEquals(new float[] { 3, 5, 7, 9 }, nd.getColumn(1).toFloatVector(), 0F);
		nd.muli(5);
		Assert.assertArrayEquals(new float[] { 40, 45 }, nd.getRow(3).toFloatVector(), 0F);
		nd.divi(2);
		Assert.assertArrayEquals(new float[] { 5, 10, 15, 20 }, nd.getColumn(0).toFloatVector(), 0F);
	}
	
	public @Test void testVectorOperations() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
		INDArray column = Nd4j.create(new float[] { 5, 6 }, new int[] { 2, 1 });
		INDArray row = Nd4j.create(new float[] { 5, 6 }, new int[] { 2 }); // 1 * 2
		nd.addiColumnVector(column);
		Assert.assertArrayEquals(new float[] { 6, 7 }, nd.getRow(0).toFloatVector(), 0F);
		nd.addiRowVector(row);
		Assert.assertArrayEquals(new float[] { 13, 16 }, nd.getColumn(1).toFloatVector(), 0F);
	}
	
	public @Test void testMatrixOperations() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
		INDArray nd2 = Nd4j.create(new float[] { 5, 6, 7, 8 }, new int[] { 2, 2 });
		nd.addi(nd2);
		Assert.assertArrayEquals(new float[] { 10, 12 }, nd.getRow(1).toFloatVector(), 0F);
		nd.muli(nd2);
		Assert.assertArrayEquals(new float[] { 30, 70 }, nd.getColumn(0).toFloatVector(), 0F);
	}
	
	public @Test void testInnerProduct() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new int[] { 4, 2 }); // 4 * 2
		INDArray row = Nd4j.create(new float[] { 1, 2 }, new int[] { 2 }); // 1 * 2
		INDArray column = Nd4j.create(new float[] { 3, 4 }, new int[] { 2, 1 }); // 2 * 1
		INDArray scalar = row.mmul(column);
		Assert.assertArrayEquals(new int[] { 1, 1 }, scalar.shape());
		Assert.assertThat(scalar.getFloat(0, 0), Matchers.is(11F));
		INDArray nd2 = nd.mmul(column);
		Assert.assertArrayEquals(new int[] { 4, 1 }, nd2.shape());
		Assert.assertArrayEquals(new float[] { 11, 25, 39, 53 }, nd2.getColumn(0).toFloatVector(), 0F);
	}
	
	public @Test void testOuterProduct() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new int[] { 4, 2 }); // 4 * 2
		INDArray row = Nd4j.create(new float[] { 1, 2 }, new int[] { 1, 2 }); // 1 * 2
		INDArray column = Nd4j.create(new float[] { 3, 4 }, new int[] { 2, 1 }); // 2 * 1
		INDArray matrix = column.mmul(row);
		Assert.assertArrayEquals(new int[] { 2, 2 }, matrix.shape());
		Assert.assertArrayEquals(new float[] { 6, 8 }, matrix.getColumn(1).toFloatVector(), 0F);
		INDArray nd2 = nd.mmul(matrix);
		Assert.assertArrayEquals(new int[] { 4, 2 }, nd2.shape());
		Assert.assertArrayEquals(new float[] { 22, 50, 78, 106 }, nd2.getColumn(1).toFloatVector(), 0F);
	}
	
	public @Test void testTranspose() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new int[] { 4, 2 }); // 4 * 2
		INDArray transpose = nd.transpose();
		Assert.assertArrayEquals(new int[] { 2, 4 }, transpose.shape());
		Assert.assertArrayEquals(new float[] { 5, 6 }, transpose.getColumn(2).toFloatVector(), 0F);
	}
	
	public @Test void testReshape() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new int[] { 4, 3 }); // 4 * 3
		INDArray nd2 = nd.reshape(6, 2);
		Assert.assertArrayEquals(new int[] { 6, 2 }, nd2.shape());
		Assert.assertArrayEquals(new float[] { 9, 10 }, nd2.getRow(4).toFloatVector(), 0F);
	}
	
	public @Test void testLinearView() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new int[] { 4, 3 }); // 4 * 3
		INDArray nd2 = nd.linearView();
		Assert.assertArrayEquals(new int[] { 1, 12 }, nd2.shape());
		Assert.assertArrayEquals(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, nd2.getRow(0).toFloatVector(), 0F);
//		INDArray nd3 = nd.linearViewColumnOrder();
//		Assert.assertArrayEquals(new int[] { 1, 12 }, nd3.shape());
//		Assert.assertArrayEquals(new float[] { 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12 }, nd3.getRow(0).toFloatVector(), 0F);
	}
	
	public @Test void testBrodercast() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
		INDArray nd2 = nd.broadcast(new int[] { 3, 12 });
		Assert.assertArrayEquals(new int[] { 3, 12 }, nd2.shape());
		Assert.assertArrayEquals(new float[] { 5, 5, 5 }, nd2.getColumn(4).toFloatVector(), 0F);
		
		nd2.put(2, 11, 111);
		INDArray nd3 = nd2.broadcast(1, 12);
		Assert.assertThat(nd3.shape(), Matchers.is(new int[] { 1, 12 }));
	}
	
	public @Test void testFunctions() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new int[] { 3, 4 });
		INDArray nd2 = Transforms.sigmoid(nd);
		Assert.assertThat(nd2, NDMatchers.lessThan(1F));
		Assert.assertThat(nd2, NDMatchers.generateThan(0F));
		Assert.assertThat(nd2, NDMatchers.increasing());
		
		INDArray nd3 = Transforms.tanh(nd);
		Assert.assertThat(nd3, NDMatchers.lessThan(1F));
		Assert.assertThat(nd3, NDMatchers.generateThan(-1F));
	}
	
	public @Test void testNormalized() {
		INDArray nd = Nd4j.create(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new int[] { 3, 4 });
		INDArray nd2 = Transforms.normalizeZeroMeanAndUnitVariance(nd);
		System.out.println(nd2);
		INDArray nd3 = nd2.gte(1.0f);
		System.out.println(nd3);
	}
}
