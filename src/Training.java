import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Training {
    public static void main(String[] args) throws FileNotFoundException {
        List<Example> examples = new ArrayList<>();
        examples = readFile(args[0]); //examples
        int nsteps = Integer.parseInt(args[1]); //steps
        double alpha = Double.parseDouble(args[2]); //

        PerceptronClassifier p_classifier = new PerceptronClassifier(examples.get(0).inputs.length);

        if (alpha > 0) {
            p_classifier.train(examples, nsteps, alpha);
        } else {
            p_classifier.train(examples, 100000, new LearningRateSchedule() {
                public double alpha(int t) { return 1000.0/(1000.0+t); }
            });
        }
    }

    public static List<Example> readFile(String filename) throws FileNotFoundException {
        Scanner file_in = new Scanner(new File(filename));
        List<Example> examples = new ArrayList<>();

        while(file_in.hasNext()) {
            String line = file_in.nextLine();
            String[] new_line = line.split(",");
//            for (String i : new_line) {
//                System.out.println(i);
//            }
            Example e = new Example(3); //initializing Example for input from text
            e.inputs[0] = 1;
            e.inputs[1] = Double.parseDouble(new_line[0]);
            e.inputs[2] = Double.parseDouble(new_line[1]);
            e.output = Integer.parseInt(new_line[2]);
            examples.add(e);
        }

        return examples; //returning examples in the text file
    }
}
