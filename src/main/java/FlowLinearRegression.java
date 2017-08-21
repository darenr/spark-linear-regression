import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class FlowLinearRegression implements TrainFlowAlgorithmInterface {

	protected final Logger LOGGER = Logger.getLogger(FlowLinearRegression.class.getName());

	public static final String FEATURE_NAME = "_features";
	public static final String PREDICTION_NAME = "_prediction";

	private final LinearRegression lr = new LinearRegression();

	@Override
	public DataFrame execute(DataFrame df, TrainingOptions opts) throws Exception {

		ParamMap[] paramGrid = new ParamGridBuilder().addGrid(lr.regParam(), new double[] { 0.1, 0.01 })
				.addGrid(lr.fitIntercept()).addGrid(lr.elasticNetParam(), new double[] { 0.0, 0.5, 1.0 }).build();

		DataFrame dataframe = df.withColumn(opts.getTargetVariable(),
				df.col(opts.getTargetVariable()).cast(DataTypes.DoubleType));

		List<PipelineStage> stages = featureColPipelineStages(dataframe.schema(), opts.getFeatureVariables());
		Pipeline pipeline = new Pipeline().setStages(stages.toArray(new PipelineStage[stages.size()]));
		PipelineModel featurizationModel = pipeline.fit(dataframe);
		DataFrame featurizedDf = featurizationModel.transform(dataframe);

		DataFrame[] splits = featurizedDf.randomSplit(new double[] { 0.9, 0.1 }, 42);
		DataFrame training = splits[0];
		DataFrame test = splits[1];

		TrainValidationSplit tvs = new TrainValidationSplit()
				.setEstimator(lr.setLabelCol(opts.getTargetVariable()).setFeaturesCol(FEATURE_NAME)
						.setPredictionCol(PREDICTION_NAME))
				.setEvaluator(new RegressionEvaluator().setLabelCol(opts.getTargetVariable())
						.setPredictionCol(PREDICTION_NAME))
				.setEstimatorParamMaps(paramGrid).setTrainRatio(opts.getTrainFactor());

		TrainValidationSplitModel tvsmodel = tvs.fit(training);
		LinearRegressionModel model = (LinearRegressionModel) tvsmodel.bestModel();

		if (opts.getAutoSave()) {
			model.save("/tmp/models/" + opts.getModelName());
		}

		DataFrame scoredDf = model.transform(test).withColumnRenamed(PREDICTION_NAME, "Predicted")
				.withColumnRenamed(opts.getTargetVariable(), "Actual");

		RegressionMetrics metrics = new RegressionMetrics(scoredDf.select("Predicted", "Actual"));

		LOGGER.info("meanAbsoluteError: " + metrics.meanAbsoluteError());
		LOGGER.info("meanSquaredError: " + metrics.meanSquaredError());
		LOGGER.info("r2: " + metrics.r2());

		return scoredDf;

	}

	public List<PipelineStage> featureColPipelineStages(StructType schema, List<String> features) throws Exception {
		List<PipelineStage> stages = new ArrayList<>();
		List<String> accumulator = new ArrayList<>();

		for (String featureName : features) {
			final String featureTypeName = schema.apply(featureName).dataType().simpleString();
			switch (featureTypeName) {
			case "double": {
				// good - this is expected
				accumulator.add(featureName);
				break;
			}
			default: {
				throw new Exception(String.format("feature(%s) has datatype (%s) which has no encoding for.",
						featureName, featureTypeName));
			}
			}

		}

		VectorAssembler assembler = new VectorAssembler().setInputCols(features.toArray(new String[features.size()]))
				.setOutputCol(FEATURE_NAME);
		stages.add(assembler);

		return stages;
	}

	@Override
	public String getName() {
		return lr.toString();
	}
}