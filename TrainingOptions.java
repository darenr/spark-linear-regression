import java.util.ArrayList;
import java.util.List;

public class TrainingOptions {
	private double testFactor = 0.1;
	private String target;
	private List<String> features = new ArrayList<String>();

	/*
	 * the fraction of training data held out to for testing against
	 */
	public TrainingOptions setTestFactor(double testFactor) {
		this.testFactor = testFactor;
		return this;
	}

	/*
	 * the target variable name
	 */
	public TrainingOptions setTargetVariable(final String target) {
		this.target = target;
		return this;
	}

	public String getTargetVariable() {
		return target;
	}

	/*
	 * add feature variable by name
	 */
	public TrainingOptions addFeatureVariables(final String... features) {
		for (String feature : features) {
			this.features.add(feature);
		}
		return this;
	}

	public List<String> getFeatureVariables() {
		return features;
	}

	public double getTestFactor() {
		return this.testFactor;
	}

	public double getTrainFactor() {
		return 1.0 - this.testFactor;
	}
}
