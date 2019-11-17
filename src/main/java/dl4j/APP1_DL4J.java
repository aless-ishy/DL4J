package dl4j;

import java.io.*;
import java.util.ArrayList;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class APP1_DL4J {

    static final int NUMBER_PIXELS = 784;
    static int nodes_nLh = 100;
    static double learning_rate = 0.2;
    static int epochs_total = 1;
    static int random_seed = 123;
    static int batchSize = 32;


    public static void main(String[] args) throws IOException, InterruptedException {
        MultiLayerConfiguration configuration = configureNeuralNetwork();
        MultiLayerNetwork neuralNetwork = createNeuralNetwork(configuration);
        DataSetIterator dataSetIterator = readCSV_File("data/mnist_train.csv");
        trainNeuralNetwork(neuralNetwork,dataSetIterator);
        DataSetIterator testIterator = readCSV_File("data/mnist_test.csv");
        evaluateNeuralNetwork(neuralNetwork,testIterator);
    }


    private static void evaluateNeuralNetwork(MultiLayerNetwork neuralNetwork, DataSetIterator testIterator) throws IOException {
        Evaluation score = new Evaluation(10);
        while (testIterator.hasNext()) {
            DataSet next_data = testIterator.next();
            INDArray output = neuralNetwork.output(next_data.getFeatures());
            score.eval(next_data.getLabels(), output);
        }

        System.out.println(score);

    }

    private static void trainNeuralNetwork(MultiLayerNetwork neuralNetwork, DataSetIterator dataSetIterator) {
        for (int i = 0; i < epochs_total; i++)
            neuralNetwork.fit(dataSetIterator);
    }


    private static DataSetIterator readCSV_File(String path) throws IOException, InterruptedException {
        File file = new File(path);
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(file));

        return new RecordReaderDataSetIterator(recordReader, batchSize, 0, 10);

    }

    private static MultiLayerNetwork createNeuralNetwork(MultiLayerConfiguration configuration) {
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(1));
        return model;
    }

    private static MultiLayerConfiguration configureNeuralNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(random_seed)
                .updater(new Sgd(learning_rate))
                .list()
                .layer(0, configureWih())
                .layer(1, configureWho())
                .pretrain(false)
                .backprop(true)
                .build();
        return conf;

    }

    private static Layer configureWih() {

        DenseLayer layer = new DenseLayer.Builder()
                .nIn(NUMBER_PIXELS)
                .nOut(nodes_nLh)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.NORMAL)
                .build();
        return layer;
    }


    private static Layer configureWho() {
        OutputLayer layer = new OutputLayer.Builder(LossFunction.MSE)
                .nIn(nodes_nLh)
                .nOut(10)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.NORMAL)
                .build();
        return layer;
    }

}

