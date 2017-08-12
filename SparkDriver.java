import java.util.Arrays;
import java.util.logging.Logger;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

public class SparkDriver {
	public static void main(String[] args) throws Exception {

		Logger LOGGER = Logger.getLogger(SparkDriver.class.getName());

		if (args.length < 2) {
			System.err.println("USAGE: <op> ...");
			System.exit(-1);
		}

		SparkConf sparkConf = new SparkConf().setAppName("ml-flow-test-driver").setMaster("local");
		SparkContext sc = new SparkContext(sparkConf);
		SQLContext sqlContext = SQLContext.getOrCreate(sc);

		final String op = args[0];
		switch (op.toLowerCase()) {
		case "train": {

			if (args.length < 6) {
				System.err.println("USAGE: <op> <algorithm> <json data path> <target> <features...>");
				System.exit(-1);
			}

			final String implementationName = args[1];
			final String jsonDataPath = args[2];
			final String targetVariable = args[3];
			final String[] featureVariables = Arrays.copyOfRange(args, 4, args.length);

			TrainingOptions opts = new TrainingOptions();

			opts.setTargetVariable(targetVariable).addFeatureVariables(featureVariables);
			opts.setAutoSave(true).setModelName("test");

			/*
			 * try to load the data
			 */
			DataFrame dfInput = sqlContext.read().json(jsonDataPath);

			LOGGER.info("input dataframe has " + dfInput.count() + " rows");

			TrainFlowAlgorithmInterface tf = TrainerFactory.makeTrainer(implementationName);

			DataFrame dfOutput = tf.execute(dfInput, opts);

			dfOutput.show();

			break;
		}
		}

	}

}
