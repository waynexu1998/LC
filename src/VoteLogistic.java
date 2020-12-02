import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class VoteLogistic {
    public static void main(String[] args) throws FileNotFoundException {
        List<Example> examples = new ArrayList<>();
        examples = readFile(args[0]); //examples
        int nsteps = Integer.parseInt(args[1]); //steps
        double alpha = Double.parseDouble(args[2]); //

        LogisticClassifier l_classifier = new LogisticClassifier(examples.get(0).inputs.length);

        if (alpha > 0) {
            l_classifier.train(examples, nsteps, alpha);
        } else {
            l_classifier.train(examples, 100000, new LearningRateSchedule() {
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
            Example e = new Example(17); //initializing Example for input from text
            e.inputs[0] = 1;
            e.inputs[1] = Integer.parseInt(new_line[0]);
            e.inputs[2] = Integer.parseInt(new_line[1]);
            e.inputs[3] = Integer.parseInt(new_line[2]);
            e.inputs[4] = Integer.parseInt(new_line[3]);
            e.inputs[5] = Integer.parseInt(new_line[4]);
            e.inputs[6] = Integer.parseInt(new_line[5]);
            e.inputs[7] = Integer.parseInt(new_line[6]);
            e.inputs[8] = Integer.parseInt(new_line[7]);
            e.inputs[9] = Integer.parseInt(new_line[8]);
            e.inputs[10] = Integer.parseInt(new_line[9]);
            e.inputs[11] = Integer.parseInt(new_line[10]);
            e.inputs[12] = Integer.parseInt(new_line[11]);
            e.inputs[13] = Integer.parseInt(new_line[12]);
            e.inputs[14] = Integer.parseInt(new_line[13]);
            e.inputs[15] = Integer.parseInt(new_line[14]);
            e.inputs[16] = Integer.parseInt(new_line[15]);
            e.output = Integer.parseInt(new_line[16]);
            examples.add(e);
        }

        return examples; //returning examples in the text file
    }
}
