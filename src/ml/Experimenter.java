/*Trevor Truog
 * Machine Learning Homework 2
 * Experimenter.java: analyzes data set stored in .csv file
 * 23 September 2013
 * Note: part 4 write-up is in bin
 */

//imports
package ml;
import java.util.ArrayList;
import java.util.Vector;
import ml.classifiers.*;

//contains program's main method
public class Experimenter {
	
	//user-set depth limit
	int depthLimit = 5;
	
	public static void main(String[] args){
		
		//instantiate class and get data set from .csv
		Experimenter exp = new Experimenter();
		DataSet dataSet = new DataSet("bin/titanic-train.csv", 6);
		
		//print out average accuracy of trained and random model
		System.out.println("Average Accuracy using trained model: " + 
				exp.averageAccuracy(dataSet) +
				"\nAverage Accuracy using random model: " +
				exp.averageRandomAccuracy(dataSet));
	}

	//calculate average accuracy for trained model
	public Double averageAccuracy(DataSet dataSet){
		
		//instantiate structures
		DecisionTreeClassifier tree = new DecisionTreeClassifier();
		Vector<Double> accuracyVector = new Vector<Double>();
		
		//do calculation 100 times
		for(int i = 0; i<100;i++){
			
			//split the data into training and test data
			DataSet[] splits = dataSet.split(0.80);
			tree.setDepthLimit(this.depthLimit);
			tree.train(splits[0]);//train tree
			
			//get test data
			ArrayList<Example> testData = splits[1].getData();
			
			//variables for:
			//predicted label, actual label, ratio of matches between the two
			Double predictLabel;
			Double actualLabel;
			int matches = 0; int totalEntries = 0;
			
			//count total number of matches
			for(Example e: testData){
				totalEntries++;
				predictLabel = tree.classify(e); 
				actualLabel = e.getLabel();
				if(predictLabel.compareTo(actualLabel)==0){
					matches++;
				}
			}
			//get matches/entries ratio
			double accuracy = (double)matches/(double)totalEntries;
			accuracyVector.add(accuracy);
		}
		
		//calculate average of 100 test runs and return
		Double sum = 0.0;
		for(Double d: accuracyVector){
			sum+=d;
		}
		
		Double averageAccuracy = sum/accuracyVector.size();
		return averageAccuracy;
	}
	
	//calculate average accuracy using the random model
	public Double averageRandomAccuracy(DataSet dataSet){
	
		RandomClassifier rand = new RandomClassifier();//instantiate structures
		Vector<Double> accuracyVector = new Vector<Double>();
		
			//split data and get test data
			DataSet[] splits = dataSet.split(0.8);
			ArrayList<Example> testData = splits[0].getData();
			
			//do same as in averageAccuracy()
			Double predictLabel;
			Double actualLabel;
			int matches = 0; int totalEntries = 0;
			for(Example e: testData){
				
				totalEntries++;
				predictLabel = rand.classify(e);
				actualLabel = e.getLabel();
				
				if(predictLabel.compareTo(actualLabel)==0){
					matches++;
				}
			}
			
			double accuracy = (double)matches/(double)totalEntries;
			accuracyVector.add(accuracy);
		
		Double sum = 0.0;
		Double averageAccuracy;
		
		for(Double d : accuracyVector){
			sum += d;
		}
		
		averageAccuracy = sum / accuracyVector.size();
		return averageAccuracy;
	}
}
