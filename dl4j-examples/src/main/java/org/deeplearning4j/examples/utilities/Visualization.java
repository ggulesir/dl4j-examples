package org.deeplearning4j.examples.utilities;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.util.List;

import static org.jfree.chart.ChartFactory.createXYLineChart;

/**
 * Created by gizem on 4/22/17.
 */
public class Visualization {
    /**
     * Generate an xy plot of the datasets provided.
     */
    public static void plotDataset(XYSeriesCollection c, String title, String xAxisLabel, String yAxisLabel, String frameTitle) {

        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle(frameTitle);
        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }


    public static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nCols = (int) data.lengthLong();
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nCols; i++) {
            series.add(i + offset, data.getDouble(i));
        }

        seriesCollection.addSeries(series);

        return seriesCollection;
    }

    public static INDArray MeanOfDataset (List<INDArray> list, int rows, int columns, String name, String xAxisLabel, String yAxisLabel, String frameTitle) {

        // Initialize matrix with zeroes.
        INDArray array = Nd4j.zeros(rows,columns);

        // Create matrix stacking each time series vertically
        for(INDArray data : list){
            array = Nd4j.vstack(data,array);
        }

        // Summation along the dimension 0 (columns)
        INDArray sumOfColumns = array.sum(0);

        // Divide each element by total number of rows to find the avarage of all time series data
        for( int i=0; i<sumOfColumns.length(); i++ ){
            double val = sumOfColumns.getDouble(0,i);
            sumOfColumns.putScalar(0,i, val/rows);
        }

        // plotting the mean series
        XYSeriesCollection collection = new XYSeriesCollection();
        Visualization.createSeries(collection, sumOfColumns, 0, "Mean of " + name);
        Visualization.plotDataset(collection, name, xAxisLabel, yAxisLabel, frameTitle);
        return sumOfColumns;
    }

    public static INDArray MeanOfDataset (XYSeriesCollection collection, List<INDArray> list, int rows, int columns, String name) {

        // Initialize matrix with zeroes.
        INDArray array = Nd4j.zeros(rows,columns);

        // Create matrix stacking each time series vertically
        for(INDArray data : list){
            array = Nd4j.vstack(data,array);
        }

        // Summation along the dimension 0 (columns)
        INDArray sumOfColumns = array.sum(0);

        // Divide each element by total number of rows to find the avarage of all time series data
        for( int i=0; i<sumOfColumns.length(); i++ ){
            double val = sumOfColumns.getDouble(0,i);
            sumOfColumns.putScalar(0,i, val/rows);
        }

        // plotting the mean series
        Visualization.createSeries(collection, sumOfColumns, 0, "Mean of " + name);
        return sumOfColumns;
    }

}
