

COMPILATION INSTRUCTIONS AND EXPLANATION OF OUTPUT

————————————————

To compile from command line, enter the below command (from root of mlhw2 directory):

javac -d bin/ -cp src src/ml/Experimenter.java

warnings may appear.

to run program enter the below command:

java -cp bin ml.Experimenter

Result similar to below output should appear:

Average Accuracy using trained model: 0.7811888111888112
Average Accuracy using random model: 0.4886164623467601

The accuracy represents  the percent of examples correctly classified.
Each example consists of information on a passenger of the Titanic.
The program is predicting whether or not that passenger survived based on this information.
The output shows that the decision tree classifier correctly classified ~78% of the examples, while a random classifier correctly classified ~49% of examples.

————————————————

EXPLANATION OF DECISION TREES

————————————————

At a high level, the decision tree is picking a feature (say ‘firstClass’, that is, whether or not the passenger was in first class) from a set of examples (say Titanic passengers) and using that feature to predict the label (whether or not the passenger survived). We pick the best feature in terms of predictive accuracy out of the set of remaining features, divide the examples into two groups based on their value for that feature, and recurse, choosing a new feature to divide on. When a base case is hit, two leaf nodes representing predictions are added to the last internal node. When the tree is complete, given a test example whose label we are trying to predict, we can use the values of that example’s features to follow a path down to a leaf node and return the leaf’s prediction of whether or not that passenger survived or died, and pass that prediction back up the tree. Thus, the decision tree has a recursive ‘divide and conquer’ structure where the feature value at each node is chosen greedily.

How do we decide which feature to use? Out of the remaining examples, pick the feature that best predicts the label. To expand on this, say we are trying to predict whether a person will go on a hike based on whether or not it is raining. Out of the training examples (say there are 20), 9 had a ‘YES’ and 11 had a ‘NO’ in the ‘isRaining’ feature column. Out of the 9 with a ‘YES,’ 7 people chose not to go on a hike and 2 chose to go on a hike. Out of the 11 with a ‘NO,’ 3 people chose not to go on a hike and 8 chose to go on a hike. Taking the majority from each group, we always predict that a person will not go on a hike when it is raining, and a person always will go on a hike when it is not raining. By doing this we will score (7+8)/(9+11) = 15/20 = 75% predictive accuracy for the feature ‘isRaining.’ We do this for all of the unused features, and greedily choose the one with the highest accuracy to be the feature at this level.

————————————————

EXPLANATION OF CODE

————————————————

I did this assignment for a machine learning course that I took. I wrote the DecisionTreeClassifier and Experimenter classes. The DecisionTreeClassifier is where all the work is (Experimenter just runs the accuracy test).

The train and trainHelper methods create the decision tree. After doing some preliminary work, the main thing we do is find the feature out of all the remaining features that has the highest accuracy. This is what the getData() function does (getData() essentially is doing exactly what is described in the second paragraph of the ‘explanation of decision trees’ section). The splitData() function then splits the examples into two groups based on the value of the chosen feature. Then, the trainHelper() checks for various base cases. If a base case is hit, we either turn the current node into a leaf node or assign two leaf nodes as children of the current node. If the recursive case is hit, we add the current feature to the usedFeatures array and recurse, choosing a new feature to divide the data on in the next level. The Classify() method is used for testing. It just predicts an example’s label using the decision tree.



