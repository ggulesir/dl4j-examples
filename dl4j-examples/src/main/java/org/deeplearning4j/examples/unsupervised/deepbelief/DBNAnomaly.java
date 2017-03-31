package org.deeplearning4j.examples.unsupervised.deepbelief;

    import org.datavec.api.records.reader.RecordReader;
    import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
    import org.datavec.api.split.FileSplit;
    import org.datavec.api.util.ClassPathResource;
    import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
    import org.deeplearning4j.nn.api.OptimizationAlgorithm;
    import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
    import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
    import org.deeplearning4j.nn.conf.layers.OutputLayer;
    import org.deeplearning4j.nn.conf.layers.RBM;
    import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
    import org.deeplearning4j.nn.weights.WeightInit;
    import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
    import org.nd4j.linalg.activations.Activation;
    import org.nd4j.linalg.dataset.DataSet;
    import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
    import org.nd4j.linalg.lossfunctions.LossFunctions;
    import org.slf4j.Logger;
    import org.slf4j.LoggerFactory;

/**
 * Created by gizem on 3/30/17.
 * ucarTrain.txt unlabelled car sensor time series 577 columns 60 rows total 34620 readings
 * car.txt labelled car sensor time series 0th column labels 0 to 3, # of labels 4
 * ucarTest.txt unlabelled car sensor test data
 */
public class DBNAnomaly {


    private static Logger log = LoggerFactory.getLogger(DBNAnomaly.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 60;
        final int numColumns = 577;
        int numSamples = 34620;
        int seed = 123;
        int batchSize = 60;
        int iterations = 10;
        int listenerFreq = iterations/5;

        RecordReader recordReader = new CSVNLinesSequenceRecordReader(1, 0, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource("ucarTrain.txt").getFile()));

        log.info("Load data....");
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
            .list()
            .layer(0, new RBM.Builder().nIn(numColumns).nOut(800).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(1, new RBM.Builder().nIn(800).nOut(400).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(2, new RBM.Builder().nIn(400).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(3, new RBM.Builder().nIn(100).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(4, new RBM.Builder().nIn(50).nOut(4).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //encoding stops
            .layer(5, new RBM.Builder().nIn(4).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //decoding starts
            .layer(6, new RBM.Builder().nIn(50).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(7, new RBM.Builder().nIn(100).nOut(400).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(8, new RBM.Builder().nIn(400).nOut(800).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
            .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(800).nOut(numColumns).build())
            .pretrain(true).backprop(true)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(listenerFreq));

        log.info("Train model....");
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));

        }

    }

}
