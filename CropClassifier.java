import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.io.PrintWriter;
import java.util.Random;
import java.util.Scanner;

public class CropClassifier {

    public static void main(String[] args) throws Exception {
        
        DataSource source = new DataSource("Crop_recommendation.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        int folds = 5;
        System.out.println("=== Performing " + folds + "-Fold Cross-Validation ===");

        for (int i = 0; i < folds; i++) {
            Instances train = dataset.trainCV(folds, i);
            Instances test = dataset.testCV(folds, i);

            J48 tree = new J48();
            tree.buildClassifier(train);
            

            Evaluation eval = new Evaluation(train);
           eval.crossValidateModel(tree, dataset, 5, new Random(1));
//            System.out.println(eval.toSummaryString());
//            System.out.println(eval.toClassDetailsString());
            eval.evaluateModel(tree, test);

            System.out.println("\nFold " + (i + 1) + ":");
            System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());

//            double precision = eval.weightedPrecision();
//            if (Double.isNaN(precision)) {
//                precision = 0.0;
//            }
          
            System.out.printf("Weighted Precision: %.4f\n",eval.weightedPrecision());

            System.out.printf("Weighted Recall: %.4f\n", eval.weightedRecall());

            System.out.println("Class-wise Precision:");
            for (int c = 0; c < dataset.numClasses(); c++) {
                double classPrec = eval.precision(c);
                if (Double.isNaN(classPrec)) {
                    classPrec = 0.0;
                }
                System.out.printf("  Class '%s': %.4f\n", dataset.classAttribute().value(c), classPrec);
            }
        }
        

        J48 finalTree = new J48();
        finalTree.buildClassifier(dataset);

        System.out.println("\n=== Final Decision Tree ===");
        System.out.println(finalTree.toString());

        String graph = finalTree.graph();
        try (PrintWriter out = new PrintWriter("tree.dot")) {
            out.println(graph);
        }
        System.out.println("Decision tree saved as 'tree.dot'. Use Graphviz to convert it to an image.");

        Scanner input = new Scanner(System.in);
        double[] instanceValues = new double[dataset.numAttributes()];

        String[] prompts = {"Nitrogen 👎", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "pH", "Rainfall"};
        for (int i = 0; i < dataset.numAttributes() - 1; i++) {
            System.out.print(prompts[i] + ": ");
            instanceValues[i] = input.nextDouble();
        }

        DenseInstance userInput = new DenseInstance(1.0, instanceValues);
        userInput.setDataset(dataset);

        double prediction = finalTree.classifyInstance(userInput);
        String crop = dataset.classAttribute().value((int) prediction);
        System.out.println("\nRecommended Crop: " + crop);
    }
}