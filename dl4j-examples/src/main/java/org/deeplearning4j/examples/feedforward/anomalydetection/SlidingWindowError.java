package org.deeplearning4j.examples.feedforward.anomalydetection;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utilities.Visualization;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Created by gizem on 4/25/17.
 */
public class SlidingWindowError {
    private static Logger log = LoggerFactory.getLogger(SlidingAnomaly.class);

    public static void main(String[] args) throws Exception {

        final int NUM_OF_ROWS = 60;
        final int NUMBER_OF_COLUMNS = 577;
        int windowSize = 100;
        int batchSize = 1;
        final int seed = 2457;
        int nEpochs = 10;
        INDArray window;
        double score;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.ADAGRAD)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.03)
            .regularization(true).l2(0.0001)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(windowSize).nOut(50)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(50).nOut(4)
                .build())
            .layer(2, new DenseLayer.Builder().nIn(4).nOut(50)
                .build())
            .layer(3, new OutputLayer.Builder().nIn(50).nOut(windowSize)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .pretrain(false).backprop(true)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(100));

        RecordReader recordReader = new CSVNLinesSequenceRecordReader(1, 0, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource("ucarTrain.txt").getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize);

        RecordReader uTestRecordReader = new CSVNLinesSequenceRecordReader(1, 0, ",");
        uTestRecordReader.initialize(new FileSplit(new ClassPathResource("ucarTest.txt").getFile()));
        DataSetIterator uTestIterator = new RecordReaderDataSetIterator(uTestRecordReader,batchSize);

        RecordReader testRecordReader = new CSVNLinesSequenceRecordReader(1, 0, ",");
        testRecordReader.initialize(new FileSplit(new ClassPathResource("carTest.txt").getFile()));
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSize);

        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> featuresUTest = new ArrayList<>();

        while(iterator.hasNext() && testIterator.hasNext() && uTestIterator.hasNext()){
            DataSet ds = iterator.next();
            featuresTrain.add(ds.getFeatureMatrix());
            DataSet dsTest = testIterator.next();
            featuresTest.add(dsTest.getFeatureMatrix());
            DataSet dsuTest = uTestIterator.next();
            featuresUTest.add(dsuTest.getFeatureMatrix());
        }

        //Train model:
        for( int epoch=0; epoch<nEpochs; epoch++ ){
            for(INDArray data : featuresTrain){
                // Shift the window by 1
                for (int i = 0; i<(NUMBER_OF_COLUMNS - windowSize + 1); i++) {
                    window = data.get(NDArrayIndex.interval(i,windowSize+i,false));
                    net.fit(window,window);
                }
            }
            System.out.println("Epoch " + epoch + " complete");
        }

        //Evaluate the model on the test data
        //Score each example in the test set separately
        //Compose a map that relates each digit to a list of (score, example) pairs
        INDArray errorByWindow = Nd4j.create(NUM_OF_ROWS, NUMBER_OF_COLUMNS - windowSize + 1);
        XYSeriesCollection collection = new XYSeriesCollection();
        Visualization.MeanOfDataset(collection,featuresUTest, NUM_OF_ROWS, NUMBER_OF_COLUMNS, "Test Data");
        for( int i=0; i<featuresTest.size(); i++ ){
            INDArray testData = featuresUTest.get(i);
            int nRows = testData.rows();
            for( int j=0; j<nRows; j++){
                INDArray example = testData.getRow(j);
                for (int k = 0; k<(NUMBER_OF_COLUMNS - windowSize + 1); k++) {
                    window = example.get(NDArrayIndex.interval(k,windowSize+k,false));
                    score = net.score(new DataSet(window, window));
                    errorByWindow.put(j,k, score);
                }
                Visualization.createSeries(collection, errorByWindow.getRow(j), 0, i + "th Error");
            }
        }

        for( int i=0; i<featuresTest.size(); i++ ){
            Visualization.createSeries(collection, featuresUTest.get(i), 0, i + "th Instance");
        }

        Visualization.plotDataset(collection, "Sliding Anomaly Errors", "TimeStep", "Error/Sensor Value", "Test Data");

    }
}
