import org.apache.spark.sql.DataFrame;

public interface TrainFlowAlgorithmInterface {
	public DataFrame execute(DataFrame data, TrainingOptions opts) throws Exception;

	public String getName();
}
