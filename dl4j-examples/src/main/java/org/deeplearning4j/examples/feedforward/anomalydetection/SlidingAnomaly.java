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
import org.deeplearning4j.optimize.api.IterationListener;
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
 * Anomaly Detection using simple autoencoder without pretraining
 * Using sliding window for the input data
 * This is accomplished in this example by using reconstruction error: stereotypical
 * examples should have low reconstruction error, whereas outliers should have high
 * reconstruction error. All datasets has same size and dimensions
 *       577 columns 60 rows total 34620 readings
 *
 * ucarTrain.txt unlabelled car sensor training time series
 *
 * carTest.txt (for testing) labelled car sensor time series
 *      0th column corresponds to label in range [0,3]
 *      total # of labels is 4
 *
 * ucarTest.txt unlabelled car sensor test data
 * Created by gizem on 4/17/17.
 */
public class SlidingAnomaly {

    private static Logger log = LoggerFactory.getLogger(SlidingAnomaly.class);

    public static void main(String[] args) throws Exception {

        final int NUM_OF_ROWS = 60;
        final int NUMBER_OF_COLUMNS = 577;
        int windowSize = 100;
        int batchSize = 1;
        final int seed = 2457;
        int nEpochs = 1;
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

        // Create simple array for printing out best & worst scores from each class type
        INDArray arrayLabels = Nd4j.create(1,NUM_OF_ROWS);

        //Evaluate the model on the test data
        //Score each example in the test set separately
        //Compose a map that relates each digit to a list of (score, example) pairs
        //Then find N best and N worst scores per digit
        Map<Integer,List<Pair<Double,INDArray>>> listsByLabel = new HashMap<>();
        for( int i=0; i<4; i++ ) listsByLabel.put(i,new ArrayList<>());

        for( int i=0; i<featuresTest.size(); i++ ){
            INDArray testData4Label = featuresTest.get(i);
            INDArray testData = featuresUTest.get(i);
            int nRows = testData4Label.rows();
            for( int j=0; j<nRows; j++){
                INDArray example4Label = testData4Label.getRow(j);
                int label = (int) example4Label.getDouble(j,0);
                INDArray example = testData.getRow(j);
                arrayLabels.add(label);
                for (int k = 0; k<(NUMBER_OF_COLUMNS - windowSize + 1); k++) {
                    window = example.get(NDArrayIndex.interval(k,windowSize+k,false));
                    score = net.score(new DataSet(window, window));
                    // Add (score, example) pair to the appropriate list
                    List allPairs = listsByLabel.get(label);
                    allPairs.add(new ImmutablePair<>(score, window));
                }

            }
        }

        //Sort each list in the map by score
        Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                return Double.compare(o1.getLeft(),o2.getLeft());
            }
        };

        for(List<Pair<Double, INDArray>> allPairs : listsByLabel.values()){
            Collections.sort(allPairs, c);
        }

        //After sorting, select N best and N worst scores (by reconstruction error) for each class, where N=5
        List<INDArray> best = new ArrayList<>(20);
        List<INDArray> worst = new ArrayList<>(20);
        for( int i=0; i<4; i++ ){
            List<Pair<Double,INDArray>> list = listsByLabel.get(i);
            for( int j=0; j<5; j++ ){
                best.add(list.get(j).getRight());
                worst.add(list.get(list.size()-j-1).getRight());
            }
            System.out.println("Best score from class " + i + " : " + list.get(0).getLeft());
            System.out.println("Worst score from class " + i + ": " + list.get(list.size()-1).getLeft());
        }

        XYSeriesCollection collection = new XYSeriesCollection();
        Visualization.createSeries(collection, best.get(0), 0, "Best");
        Visualization.createSeries(collection, best.get(1), 0, "2nd Best");
        Visualization.createSeries(collection, worst.get(0), 0, "Worst");
        Visualization.createSeries(collection, worst.get(1), 0, "2nd Worst");

        Visualization.plotDataset(collection,"Sliding Anomaly", "TimeStep", "Sensor Readings", "Test Data");

    }
}
