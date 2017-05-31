package org.deeplearning4j.examples.recurrent.seqclassification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by gizem on 5/16/17.
 */
public class SimpleSequenceClassification {
    private static final Logger log = LoggerFactory.getLogger(SimpleSequenceClassification.class);


    public static void main(String[] args) throws Exception {

        final int NUM_OF_ROWS = 60;
        final int NUMBER_OF_COLUMNS = 577;
        int miniBatchSize = NUMBER_OF_COLUMNS;
        int numLabelClasses = 4;
        int batchSize = 1;

        RecordReader train = new CSVNLinesSequenceRecordReader(1, 0, ",");
        train.initialize(new FileSplit(new ClassPathResource("trainSequenceClassification.txt").getFile()));
        DataSetIterator trainIterator = new RecordReaderDataSetIterator(train, batchSize, 0, numLabelClasses);

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIterator);
        trainIterator.reset();
        trainIterator.setPreProcessor(normalizer);

        RecordReader test = new CSVNLinesSequenceRecordReader(1, 0, ",");
        test.initialize(new FileSplit(new ClassPathResource("testSequenceClassification.txt").getFile()));
        DataSetIterator testIterator = new RecordReaderDataSetIterator(test, batchSize, 0, numLabelClasses);

        //Note that we are using the exact same normalization process as the training data
        testIterator.setPreProcessor(normalizer);


        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(0.005)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(NUMBER_OF_COLUMNS).nOut(NUMBER_OF_COLUMNS).build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nIn(NUMBER_OF_COLUMNS).nOut(numLabelClasses).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 15;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIterator);

            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(testIterator);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testIterator.reset();
            trainIterator.reset();
        }

        log.info("----- Example Complete -----");
    }
}
