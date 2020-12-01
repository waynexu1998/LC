public class LogisticClassifier extends LinearClassifier {
	
	public LogisticClassifier(double[] weights) {
		super(weights);
	}
	
	public LogisticClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * A LogisticClassifier uses the logistic update rule
	 * (AIMA Eq. 18.8): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
		// This must be implemented by you
		for(int i=0; i<this.weights.length; i++) {
			double h_w = threshold(VectorOps.dot(this.weights, x));
			this.weights[i] = this.weights[i] + alpha*(y - h_w) * x[i] * h_w * (1-h_w);
		}
	}
	
	/**
	 * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
	 */
	public double threshold(double z) {
		// This must be implemented by you

		//Page 726
		return 1/(1 + Math.exp(-z));
	}

}
