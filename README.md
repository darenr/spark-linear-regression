### Using Spark, run a regression model training using LinearRegression

The code is adaptable to any Regression algorithm, this is just an example.

### How to Run

You'll need Spark 1.6

```
bin/spark-submit --master local spark-linear-regression.jar train LinearRegression /home/drace/github/spark-linear-regression/advertising.json Sales Newspaper Radio TV
```
