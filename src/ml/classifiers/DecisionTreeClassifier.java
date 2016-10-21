/*Trevor Truog
 * Machine Learning Homework2
 * DecisionTreeClassifier.java: creates a decision tree
 * 23 September 2013
 */

//imports
package ml.classifiers;
import ml.DataSet;
import ml.Example;
import java.util.ArrayList;
import java.util.Vector;

//all methods to classify a decision tree
public class DecisionTreeClassifier implements Classifier {
	
	int numberOfFeatures;//number of features in data set
	DecisionTreeNode tree; //root node of tree
	int depthLimit; //depthLimit variable

	//Set the depth limit
	public void setDepthLimit(int depthLimit){
		this.depthLimit = depthLimit;
	}

	//create a decision tree using the passed in data
	public void train(DataSet data) {
		// Initial training data set
		ArrayList<Example> originalData = data.getData();
		// Vector tracking used features
		Vector<Integer> usedFeatures = new Vector<Integer>();
		//number of features used to predict label
		numberOfFeatures = data.getFeatureMap().size();
		//Predict a 0 Depth tree
		double justLeafPred = depthZeroPredict(originalData);
		//set tree equal to the root node of the decision tree,
		//which is constructed recursively with trainHelper
		tree = trainHelper(originalData, 0, usedFeatures, justLeafPred, depthLimit);
		
	}
	
	//Helper function for train method
	//Takes array of examples, next level of the tree, a vector of used features,
	//the prediction of the parent, and the depth limit as parameters
	public DecisionTreeNode trainHelper(ArrayList<Example> exampleArray, int level,
			Vector<Integer> usedFeatures, double parentPredict, int depthLimit){
		
		int nextLevel = level+1; //increment level
		int maxLevel = numberOfFeatures-1; //depth at which recursion stops
		
		//Collect featureVal and predictions for featureVal
		Vector<Double> accuracyInfo = getData(exampleArray, usedFeatures, level);
		
		int featureVal = accuracyInfo.get(2).intValue();
		
		//values given to potential leaf
		double leftPredict = accuracyInfo.get(0); 
		double rightPredict = accuracyInfo.get(1);
		
		usedFeatures.add(featureVal);//add feature to usedFeatures
		
		//dataSet split based on featureVal
		Vector<ArrayList<Example>> splitData = splitDataSet(exampleArray, featureVal);
		
		//instantiate node described in this specific recursive call
		DecisionTreeNode newNode = new DecisionTreeNode(featureVal);
		
		//booleans used for base case checks
		boolean allSameFeature = allSameFeature(exampleArray);
		boolean allSameLabel = allSameLabel(exampleArray);
		
		
		//(additional) base case used to hand a tree of Depth 0
		if(depthLimit==0){
			//create new node predicting majority label
			newNode = new DecisionTreeNode(parentPredict);
		}
		
		else if (exampleArray.size()==0) { //base case: no examples remain
			//create leaf based on what parent would have predicted
			newNode = new DecisionTreeNode(parentPredict);
	
		} else if(allSameLabel) { //base case: all labels are the same
			//create leaf predicting label
			return newNode = new DecisionTreeNode(exampleArray.get(0).getLabel());
			
		//base case: all features used or reached user-set depth limit
		} else if((level==maxLevel) || (level==depthLimit-1)) { 
			//instantiate and set leaf nodes to child of this node
			DecisionTreeNode leftLeaf = new DecisionTreeNode(leftPredict);
			DecisionTreeNode rightLeaf = new DecisionTreeNode(rightPredict);
			newNode.setLeft(leftLeaf);
			newNode.setRight(rightLeaf);

		} else if(allSameFeature) { //base case: all features are the same
			//instantiate and set leaf nodes to child of this node
			DecisionTreeNode leftLeaf = new DecisionTreeNode(leftPredict);
			DecisionTreeNode rightLeaf = new DecisionTreeNode(rightPredict);
			newNode.setLeft(leftLeaf);
			newNode.setRight(rightLeaf);

		} else { //recursive case
			newNode.setLeft(trainHelper(splitData.get(0), nextLevel, usedFeatures,
					leftPredict, depthLimit)); //create left subtree
			
			// usedFeatures is passed by reference, so as we move back up the stack
			// we need to remove the features that lower level calls added to usedFeatures
			//remove correct number of features from usedFeatures vector
			//Note: number of features removed is determined by calculating
			//the difference between the current level, and the maximum
			//depth of the tree (i.e. number of returned recursive calls since
			//last leaf node was created)
			
			int size;
			for(int i=0;i<maxLevel-level;i++){
				if(!usedFeatures.isEmpty()){
					size = usedFeatures.size()-1;
					usedFeatures.remove(size);
				} 
			}
			
			newNode.setRight(trainHelper(splitData.get(1), nextLevel, usedFeatures,
					rightPredict, depthLimit)); //create right subtree
		}
		
		return newNode; //return the created leaf or tree node
	}

	//classify Example using decision tree
	//Takes current example to analyze as a parameter
	public double classify(Example example) {
		//Clone tree so as not to manipulate its structure during analysis
		DecisionTreeNode cloneNode = tree;
		//complete cloning process
		DecisionTreeNode currentNode = cloneNode;
		//get the leaf of the tree and return its prediction
		DecisionTreeNode leaf = classifyHelper(currentNode, example);
		return leaf.prediction();
	}
	
	//helper method for classify function
	//Takes root node and current example to analyze as parameters
	public DecisionTreeNode classifyHelper(DecisionTreeNode node, Example example){
		
		//variables used for iteration
		int currentNodeFeature;
		Double currentExampleFeature;
		
		//return the node if it is a leaf
		if(node.isLeaf()){
			return node;
		}
		
		else{
			//get feature value stored in node
			currentNodeFeature = node.getFeatureIndex();
			//get the example's particular value for that feature
			currentExampleFeature = example.getFeature(currentNodeFeature);
			
			//determine whether to look in left or right subtree
			//just goes left if example's feature value is 0, right if 1
			if(currentExampleFeature.compareTo(0.0)==0){
				//return reference to left subtree
				return classifyHelper(node.getLeft(),example);	
			} else {
				//return reference to right subtree
				return classifyHelper(node.getRight(),example);	
			}
		}
	}
	
	//split the data set based on feature value
	//Takes an ArrayList of examples, and a feature value as parameters
	public Vector<ArrayList<Example>> splitDataSet(ArrayList<Example> dSet, int featureVal){
		
		//array to return
		Vector<ArrayList<Example>> returnArray = new Vector<ArrayList<Example>>();
		//set size of return array and
		//create variables used to set the size of ExampleArrays
		returnArray.setSize(2);
		int numberOfZeros = 0;
		int numberOfOnes = 0;
		
		//count if feature value = 1 or = 0
		for(Example e: dSet){
			if(e.getFeature(featureVal)==0.0) numberOfZeros++;
			if(e.getFeature(featureVal)==1.0) numberOfOnes++;
		}
		
		//data sets to be added to the returnArray
		ArrayList<Example> dSet0 = new ArrayList<Example>(numberOfZeros);
		ArrayList<Example> dSet1 = new ArrayList<Example>(numberOfOnes);
		
		//Add to data set 1 or 0 based on feature value
		for(Example e: dSet){
			if(e.getFeature(featureVal)==0.0) dSet0.add(e);
			if(e.getFeature(featureVal)==1.0) dSet1.add(e);
		}
		
		//add data sets to returnArray and return
		returnArray.set(0, dSet0);
		returnArray.set(1, dSet1);
		return returnArray;
	}
	
	//get the information necessary for tree construction
	//takes an example array, list of used features, and current level
	//of recursion as parameters. returns feature value with maximum accuracy,
	//predictions for left and right branches (assuming leaf)
	public Vector<Double> getData(ArrayList<Example> exampleArray,
			Vector<Integer> usedFeatures, int level){
		
		//Vectors for storing majority bins
		Vector<Vector<Double>> majorityBin0 = new Vector<Vector<Double>>();
		Vector<Vector<Double>> majorityBin1 = new Vector<Vector<Double>>();
		//Vector for storing accuracy values of feature predictions
		Vector<Double> accuracy = new Vector<Double>();
		
		//initialize all data collection structures
		//Add zeros to avoid null pointers
		for(int i=0;i<numberOfFeatures;i++){
			majorityBin0.add(new Vector<Double>(2));
			majorityBin1.add(new Vector<Double>(2));
			majorityBin0.get(i).add(0.0); majorityBin0.get(i).add(0.0);
			majorityBin1.get(i).add(0.0); majorityBin1.get(i).add(0.0);
			accuracy.add(0.0);
			majorityBin0.get(i).set(0,0.0); majorityBin0.get(i).set(1,0.0);
			majorityBin1.get(i).set(0,0.0); majorityBin1.get(i).set(1,0.0);
		}
		
		//Vector of doubles used for iterating feature counts
		//in majority bins
		Vector<Double> currentBinIndex;
		double binValue; 
		double currentLabel;
		
		/* Calculate: when a feature is 0, what is the count of labels that are 0 and what is the count of labels that are 1
		 * 			  when a feature is 1, what is the count of labels that are 0 and what is the count of labels that are 1
		 * More detail:
		 * MajorityBin0 holds a tuple for each feature value F, and each tuple keeps track of:
		 * when feature F is 0, what is the count of labels that are 0, and what is the count of labels that are 1
		 * 
		 * MajorityBin1 does the same for when the feature value is 1
		 * 
		 * 
		 * The double loop goes through each feature of each example and records this tally
		 */
		for(Example i : exampleArray){
			currentLabel = i.getLabel();

			for(int e=0;e<numberOfFeatures;e++){
				//if feature value is 0:
				if(i.getFeature(e)==0.0){
					currentBinIndex = majorityBin0.get(e);
					//if current label is 0:
					if(currentLabel==0.0){
						//get bin value corresponding to that feature
						binValue = currentBinIndex.get(0);
						//increment 0's bin value:
						currentBinIndex.set(0, binValue + 1);
					//else increment 1's bin value
					} else if(currentLabel==1.0){
						binValue = currentBinIndex.get(1);
						currentBinIndex.set(1, binValue + 1);
					}
					
				//repeat process described above
				} else if(i.getFeature(e)==1.0){
					currentBinIndex = majorityBin1.get(e);
					
					if(currentLabel==0.0){
						binValue = currentBinIndex.get(0);
						currentBinIndex.set(0, binValue + 1);	
					} else if(currentLabel==1.0){
						binValue = currentBinIndex.get(1);
						currentBinIndex.set(1, binValue + 1);
					}	
				}	
			}
		}
		
		//data variables used to count(bin(0 or 1) = survive/not survive)
		double bin0NotSurvived, bin0Survived, bin1NotSurvived, bin1Survived;
		// for each feature value, store a 0 if more labels were 0 than 1s, 1 otherwise
		Vector<Double> majority0 = new Vector<Double>();
		Vector<Double> majority1 = new Vector<Double>();
		
		
		/* 
		 * Calculating accuracy from majority bins. Basically asking, if I always guess one label
		 * when feature F is 0 and I always guess one (perhaps different) label when F is 1, what
		 * percentage of examples will I correctly classify?
		 * Do this for each (not already used) feature and pick the feature with the highest accuracy
		 */
		
		//loop over majority bins, getting the counts of 
		//feature-value-matches in majority bin 0 and 1
		for(int i=0;i<majorityBin0.size();i++){
			bin0NotSurvived = majorityBin0.get(i).get(0);
			bin0Survived = majorityBin0.get(i).get(1);
			bin1NotSurvived = majorityBin1.get(i).get(0);
			bin1Survived = majorityBin1.get(i).get(1);
			
			//calculate accuracy
			accuracy.set(i,(double)(Math.max(bin0NotSurvived,bin0Survived)+
					Math.max(bin1NotSurvived,bin1Survived))/exampleArray.size());
			
			//Get the max value for each feature in each majority bin
			//and add them to majority vectors
			if(bin0NotSurvived>bin0Survived) majority0.add(0.0);
			else majority0.add(1.0);
			
			if(bin1NotSurvived>bin1Survived) majority1.add(0.0);
			else majority1.add(1.0);
		}
		
		//variable used to determine the feature with the maximum accuracy
		double maxAccuracy = accuracy.get(0);
		
		//feature value with the maximum accuracy found so far
		int featureVal=0;
			
		//for each feature's accuracy:
		for(int i=1; i<accuracy.size(); i++) {
			//if that feature hasn't been used:
			if(!usedFeatures.contains(i)){
				//compare that feature's accuracy score with current highest score
				if(maxAccuracy<accuracy.get(i)) {
					//change the value of maxAccuracy
					maxAccuracy = accuracy.get(i);
					//change the feature value to the feature
					//with the maximum accuracy
					featureVal = i;
				}
			}
		}
		
		/*
		 * Now we have the feature that most accurately predicts the label out of all remaining features
		 * Return the prediction (label that resulted the majority of the time) when that feature is 0 and 1
		 * for the left and right branch respectively. If the node to be created is a leaf, the prediction
		 * will tell whether to predict survived or not survived. If the node to be created is an internal node,
		 * return the feature value with the highest accuracy to use as the predictor at this level
		 */
	
		//left and right branch prediction variables
		double leftBranch;
		double rightBranch;
		
		//if there were a majority of 0's or 1's for the feature value
		//with the maximum accuracy in bin(0 or 1) set the left and right
		//branches' prediction value to 0.0 or 1.0 accordingly
		if(majority0.get(featureVal) == 0.0) leftBranch = 0.0;
		else leftBranch = 1.0;
		if(majority1.get(featureVal) == 0.0) rightBranch = 0.0;
		else rightBranch = 1.0;
		majorityBin0.add(accuracy);
		
		//Create and return a vector containing prediction for left/right branches
		//(assuming leaf), and the feature value (assuming internal node)
		Vector<Double> trainingData = new Vector<Double>();
		trainingData.add(leftBranch);
		trainingData.add(rightBranch);
		trainingData.add((double)(featureVal));
		return trainingData;
	}
	
	//handles base case
	//base case helper function: if all examples have same label.
	// if all the examples have the same label, we simply predict this label
	public boolean allSameLabel(ArrayList<Example> exampleArray){
		Double sampleLabel;
		Double currentLabel;
		boolean allSameLabel = true;
		
		try{
			sampleLabel = exampleArray.get(0).getLabel();
			
			for(Example e: exampleArray){
				currentLabel = e.getLabel();
				if((sampleLabel.compareTo(currentLabel)!=0))allSameLabel=false;
			}
		} catch(Exception ex){}
		
		return allSameLabel;
		
	}
	
	//base case helper function: if all examples have same feature.
	// if all the features in the remaining examples have the same value, then no one feature can be better
	// than another at predicting the label, so we predict majority label in subset
	
	// handles base case
	public boolean allSameFeature(ArrayList<Example> exampleArray){
			
			boolean allSameFeature = true;
			
			try{
				Double sampleFeature = exampleArray.get(0).getFeature(0);
	
				for(Example e: exampleArray){
					 
					for(int i=0;i<numberOfFeatures;i++){
						if(e.getFeature(i)!=sampleFeature){
							allSameFeature=false;
							break;
						}
					}
				}
			} catch(Exception ex){}
			
			return allSameFeature;
	}
	
	
	//return tree (root) node
	public DecisionTreeNode getTree(DecisionTreeClassifier tree){
		return this.tree;
	}
	
	// handles edge case
	//helper function to predict the label of a 0 depth tree
	public Double depthZeroPredict(ArrayList<Example> originalData){
		
		int numberOfZeros = 0;
		int numberOfOnes = 0;
		
		
		for(Example e: originalData){
			Double label = e.getLabel();
			if(label.compareTo(0.0)==0){
				numberOfZeros++;
			} else {
				numberOfOnes++;
			}
		}
		
		if(numberOfZeros>numberOfOnes){
			return 0.0;
		} else return 1.0;
	}


}
	
	


