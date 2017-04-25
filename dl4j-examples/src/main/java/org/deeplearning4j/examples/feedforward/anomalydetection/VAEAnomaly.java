package org.deeplearning4j.examples.feedforward.anomalydetection;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

/**
 * Created by gizem on 4/1/17.
 * This example performs unsupervised anomaly detection on MNIST using a variational autoencoder, trained with a Bernoulli
 * reconstruction distribution.
 *
 * For details on the variational autoencoder, see:
 * - Kingma and Welling, 2013 - Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 *
 * For the use of VAEs for anomaly detection using reconstruction probability see:
 * - An & Cho, 2015 - Variational Autoencoder based Anomaly Detection using Reconstruction Probability
 *   http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf
 *
 *
 * Unsupervised training is performed on the entire data set at once in this example. An alternative approach would be to
 * train one model for each digit.
 *
 * After unsupervised training, examples are scored using the VAE layer (reconstruction probability). Here, we are using the
 * labels to get the examples with the highest and lowest reconstruction probabilities for each digit for plotting. In a general
 * unsupervised anomaly detection situation, these labels would not be available, and hence highest/lowest probabilities
 * for the entire data set would be used instead.
 *
 * @author Alex Black
 */
public class VAEAnomaly {

    public static void main(String[] args) throws  Exception {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 5;                    //Total number of training epochs
        int reconstructionNumSamples = 16;  //Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details
        int batchSize = 577;


        RecordReader recordReader = new CSVNLinesSequenceRecordReader(1, 0, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource("ucarTrain.txt").getFile()));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader,batchSize);

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .learningRate(0.05)
            .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
            .weightInit(WeightInit.XAVIER)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)                    //2 encoder layers, each of size 256
                .decoderLayerSizes(256, 256)                    //2 decoder layers, each of size 256
                .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                .nIn(577)                                   //Input size: 28x28
                .nOut(32)                                   //Size of the latent variable space: p(z|x) - 32 values
                .build())
            .pretrain(true).backprop(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(100));

        //Fit the data (unsupervised training)
        for( int i=0; i<nEpochs; i++ ){
            net.fit(trainIter);
            System.out.println("Finished epoch " + (i+1) + " of " + nEpochs);
        }

    }
}
