package com.lmq.ml.nd4j.test;

import org.hamcrest.Description;
import org.hamcrest.TypeSafeMatcher;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NDMatchers {
	
	public static NDOrderingComparsion lessThan(float expected) {
		return new NDOrderingComparsion(true, expected);
	}
	
	public static NDOrderingComparsion generateThan(float expected) {
		return new NDOrderingComparsion(false, expected);
	}
	
	public static NDOrdered increasing() {
		return new NDOrdered(false, true, false);
	}
	
	public static NDOrdered diminishing() {
		return new NDOrdered(true, true, false);
	}

	public static class NDOrdered extends TypeSafeMatcher<INDArray> {

		private boolean desc;
		
		private boolean row;
		
		private boolean strict;
		
		public NDOrdered(boolean desc, boolean row, boolean strict) {
			this.desc = desc;
			this.row = row;
			this.strict = strict;
		}
		
		@Override
		public void describeTo(Description description) {
			description.appendText("all value " + (strict ? "strictly " : "") + (desc ? "diminishing" : "increasing"));
		}

		@Override
		protected boolean matchesSafely(INDArray item) {
			int[] shape = item.shape();
			float pre = desc ? Float.MAX_VALUE : Float.MIN_VALUE;
			if(row) {
				for(int i = 0; i < shape[0]; i++) {
					for(int j = 0; j < shape[1]; j++) {
						float value = item.getFloat(i, j);
						if(desc ? value > pre : value < pre) {
							return false;
						}
						pre = value;
					}
				}
				return true;
			}
			for(int j = 0; j < shape[1]; j++) {
				for(int i = 0; i < shape[0]; i++) {
					float value = item.getFloat(i, j);
					if(desc ? value > pre : value < pre) {
						return false;
					}
					pre = value;
				}
			}
			return true;
		}
		
	}
	
	public static class NDOrderingComparsion extends TypeSafeMatcher<INDArray> {

		private boolean lessThan;

		private float expected;
		
		public NDOrderingComparsion(boolean lessThan, float expected) {
			this.lessThan = lessThan;
			this.expected = expected;
		}
		
		@Override
		public void describeTo(Description description) {
			description.appendText("all value " + (lessThan ? "less than " : "generate than ") + expected);
		}

		@Override
		protected boolean matchesSafely(INDArray item) {
			int[] shape = item.shape();
			for(int i = 0; i < shape[0]; i++) {
				for(int j = 0; j < shape[1]; j++) {
					float value = item.getFloat(i, j);
					if(lessThan ? value >= expected : value <= expected) {
						return false;
					}
				}
			}
			return true;
		}
		
	}
}
