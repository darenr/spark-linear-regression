
public class TrainerFactory {

	public static TrainFlowAlgorithmInterface makeTrainer(final String name) throws Exception {
		switch (name) {
		case "LinearRegression": {
			return new FlowLinearRegression();
		}
		}
		throw new Exception(String.format("No TrainFlowInterface implementation found for: ", name));
	}
}
