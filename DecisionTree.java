/**
 * Brandon Ferrell
 * Intro to Machine Learning
 * Assignment 1: Decision Tree Induction
 * 
 * Classes: DecisionTree, TreeNode
 *
 * Input: Name of the training and testing files
 * Output: Tree represented in format described in the instructions, accuracy of tree on 
 * training and test sets
 *
 * Data Structures Used: ArrayList, Custom Tree
 *
 * DecisionTree acts as the tree and the driver of the program. The class holds data points
 * read from the training and testing files and the root of the tree. The main method holds
 * all of the high-level actions outlined in the instructions.
 *
 * First, the program asks for the training and testing filenames, then the files are parsed
 * and the nevessary data is then placed into data structures. The program then builds the tree
 * from the data, prints the tree, and then prints the accuracy of the tree.
 *
 * More info on specifics of the program can be found below
 */

import java.util.*;
import java.io.*;
import java.lang.*;

class DecisionTree {


	// Attribute names in a list
	ArrayList<String> attributes = new ArrayList<>();

	// Data points of training and testing data
	ArrayList<ArrayList<Integer>> trainingInstances = new ArrayList<>();
	ArrayList<ArrayList<Integer>> testingInstances = new ArrayList<>();

	// Root of the tree
	TreeNode root;

	// Filenames given by user
	String trainingFilename;
	String testingFilename;

	public static void main(String[] args) {
		DecisionTree tree = new DecisionTree();

		try {
			tree.cliInput();
			tree.parseFiles();
			tree.buildTree();
			tree.printTree();
			tree.printAccuracy();
		} catch(Exception e) {
			System.out.println("Could not find file");
		}
	}

	/**
	* Get filenames from user
	*/
	private void cliInput() {
		Scanner in = new Scanner(System.in);

		System.out.print("Enter the name of the training file: ");
		trainingFilename = in.next();

		System.out.print("Enter the name of the testing file: ");
		testingFilename = in.next();
	}

	/**
	* Parse files and store data into necessary data structures
	*/
	private void parseFiles() throws Exception {
		FileInputStream fstream = new FileInputStream(trainingFilename);
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

		String strLine;
		boolean first = true;

		// Parse training file
		while((strLine = br.readLine()) != null) {

			// Skip blank lines
			if(strLine.trim().length() == 0) {
				continue;
			}

			if(first) {
				first = false;

				String[] splitArray = strLine.split("\\s+");
				for(String str : splitArray) {

					// Make sure not to store the class in the attributes
					if(!str.equals("class")) {
						attributes.add(str);
					}
				}
			}
			else {
				// Store data
				ArrayList<Integer> instance = new ArrayList<>();

				String[] splitArray = strLine.split("\\s+");
				for(String str : splitArray) {
					instance.add(new Integer(str));
				}

				trainingInstances.add(instance);
			}
		}

		first = true;

		fstream = new FileInputStream(testingFilename);
		br = new BufferedReader(new InputStreamReader(fstream));

		// Parsing testing file
		while((strLine = br.readLine()) != null) {

			// Skip blank lines
			if(strLine.trim().length() == 0) {
				continue;
			}

			if(first) {
				first = false;
			}
			else {
				// Store data
				ArrayList<Integer> instance = new ArrayList<>();

				String[] splitArray = strLine.split("\\s+");
				for(String str : splitArray) {
					instance.add(new Integer(str));
				}

				testingInstances.add(instance);
			}
		}

		br.close();
	}

	/**
	* Build the Decision Tree from the the data parsed from files
	*/
	private void buildTree() {

		// Create root
		root = new TreeNode();

		ArrayList<String> remainingAttrs = new ArrayList<>();

		for(String attr: attributes) {
			remainingAttrs.add(attr);
		}

		// Add the rest of the attributes the root node can use (all of them)
		root.setRemainingAttrs(remainingAttrs);

		// Add the training data points to the root and increment the +/- counters
		for(ArrayList<Integer> instance : trainingInstances) {
			if(instance.get(instance.size() - 1) == 0) {
				root.incrementClass0Count();
			}
			else {
				root.incrementClass1Count();
			}

			root.addInstance(instance);
		}

		int attrIndex;
		double bestIg;
		String igAttrName = "";
		ArrayList<TreeNode> leaves;

		while(!allLeavesPure(root) && !allAttributesUsed()) {

			// Work on all of the leaves that have attributes to use and aren't pure
			leaves = getLeaves();
			for(TreeNode node : leaves) {
				if(node.getEntropy() > 0 && node.getRemainingAttrs().size() > 0) {
					bestIg = 0;
					igAttrName = "";
					remainingAttrs = node.getRemainingAttrs();

					// Loop through each remaining attribute for the leaf node
					for(String attr : remainingAttrs) {
						attrIndex = attributes.indexOf(attr);

						// Grab the information gain for this attribute
						double ig = getInformationGain(node, attrIndex);

						// Set max ig if the new ig better
						if(ig > bestIg) {
							bestIg = ig;
							igAttrName = attr;
						}
						// If the ig is equal, pick the smaller index
						else if(ig == bestIg) {
							if(igAttrName.equals("") || attrIndex < attributes.indexOf(igAttrName)) {
								igAttrName = attr;
							}
						}
					}

					// Build nodes off leaf
					buildNewNodes(attributes.indexOf(igAttrName), igAttrName, node);
				}

				// If the leaf is pure but has remaining attributes, empty the attributes
				// We don't need them
				else if(node.getRemainingAttrs().size() > 0) {
					node.emptyAttrs();
				}
			}
					
		}
	}

	/**
	* Split a leaf node on a given attribute and add the nodes to the leaf
	*/
	private void buildNewNodes(int attrIndex, String attrName, TreeNode node) {
		TreeNode nodeZero = new TreeNode();
		TreeNode nodeOne = new TreeNode();

		// For every data point in the leaf, add each data point to its
		// correct node on attribute split
		for(ArrayList<Integer> instance : node.getInstances()) {
			if(instance.get(attrIndex) == 0) {
				if(instance.get(instance.size() - 1) == 0) {
					nodeZero.incrementClass0Count();
				}
				else {
					nodeZero.incrementClass1Count();
				}
				
				nodeZero.addInstance(instance);
			}
			else {
				if(instance.get(instance.size() - 1) == 0) {
					nodeOne.incrementClass0Count();
				}
				else {
					nodeOne.incrementClass1Count();
				}

				nodeOne.addInstance(instance);
			}
		}

		// Copy the remaining attributes from the parent to children nodes
		nodeZero.copyAttrs(node.getRemainingAttrs());
		nodeOne.copyAttrs(node.getRemainingAttrs());

		// Remove this split attribute from children. They already split on it
		nodeZero.removeAttr(attrName);
		nodeOne.removeAttr(attrName);

		// Add children to given leaf node
		node.setZeroNode(nodeZero);
		node.setOneNode(nodeOne);

		// Set the node's attribute that was split on
		node.setAttr(attrName);
	}

	/**
	* Get the information gain from a given node and attribute
	*/
	private double getInformationGain(TreeNode node, int attrIndex) {

		// Temporary nodes for calculating IG
		TreeNode testNodeZero = new TreeNode();
		TreeNode testNodeOne = new TreeNode();

		// For every data point in the leaf, add each data point to its
		// correct node on attribute split
		for(ArrayList<Integer> instance : node.getInstances()) {
			if(instance.get(attrIndex) == 0) {
				if(instance.get(instance.size() - 1) == 0) {
					testNodeZero.incrementClass0Count();
				}
				else {
					testNodeZero.incrementClass1Count();
				}
				
				testNodeZero.addInstance(instance);
			}
			else {
				if(instance.get(instance.size() - 1) == 0) {
					testNodeOne.incrementClass0Count();
				}
				else {
					testNodeOne.incrementClass1Count();
				}

				testNodeOne.addInstance(instance);
			}
		}

		// Return the information gain using conditional entropy
		return node.getEntropy() - (((double) testNodeZero.getInstances().size() / node.getInstances().size()) * testNodeZero.getEntropy() + ((double) testNodeOne.getInstances().size() / node.getInstances().size()) * testNodeOne.getEntropy());
	}

	/**
	* Get the leaves of the trees. Entry point for recursive method
	*/
	private ArrayList<TreeNode> getLeaves() {
		ArrayList<TreeNode> list = new ArrayList<>();

		getLeaves(list, root);

		return list;
	}

	/**
	* Get the leaves of the trees recursively
	*/
	private void getLeaves(ArrayList<TreeNode> list, TreeNode node) {
		if(node.isLeaf()) {
			list.add(node);
		}
		else {
			getLeaves(list, node.getZeroNode());
			getLeaves(list, node.getOneNode());
		}
	}

	/**
	* Returns whether all of the leaves in the tree are pure (0 entropy)
	*/
	private boolean allLeavesPure(TreeNode node) {
		boolean isPure = false;

		if(node.isLeaf()) {
			if(node.getEntropy() == 0) {
				isPure = true;
			}
		}
		else {
			isPure = allLeavesPure(node.getZeroNode()) && allLeavesPure(node.getOneNode());
		}

		return isPure;
	}

	/**
	* Returns whether all of the attributes have been used in the tree
	*/
	private boolean allAttributesUsed() {
		boolean used = true;

		// Get all of the leaves
		ArrayList<TreeNode> leaves = getLeaves();

		// For each leaf, test if is has any more attributes to split on
		for(TreeNode leaf : leaves) {
			if(leaf.getRemainingAttrs().size() > 0) {
				used = false;
				break;
			}
		}

		return used;
	}

	/**
	* Print the tree. Entry point for recursive method
	*/
	private void printTree() {

		// Use a string builder since Strings are immutable
		StringBuilder treePrinted = new StringBuilder();

		// Build the tree
		printTree(0, treePrinted, root);

		// Print tree to console
		System.out.println(treePrinted.toString());
	}

	/**
	* Builds the string version of the tree recursively
	*/
	private void printTree(int level, StringBuilder tree, TreeNode node) {

		// Add the indention
		for(int i = 0; i < level; i++) {
			tree.append("|  ");
		}

		// Handle for class 0
		tree.append(node.getAttr() + " = 0 :");

		if(!node.getZeroNode().isLeaf()) {
			tree.append("\n");

			// Build the left side
			printTree(level + 1, tree, node.getZeroNode());
		}
		else {

			// Get class value at leaf
			tree.append(" " + node.getZeroNode().getDominantClass() + "\n");
		}

		// Handle for class 1
		for(int i = 0; i < level; i++) {
			tree.append("|  ");
		}

		tree.append(node.getAttr() + " = 1 :");

		if(!node.getOneNode().isLeaf()) {
			tree.append("\n");

			// Build the right side
			printTree(level + 1, tree, node.getOneNode());
		}
		else {

			// Get class value at leaf
			tree.append(" " + node.getOneNode().getDominantClass() + "\n");
		}
	}

	/**
	* Calculate the accuracy of the training and testing sets and print them
	*/
	private void printAccuracy() {
		double trainingAccuracy = 0;
		double testingAccuracy = 0;
		int correct = 0;

		// Calculate the amount of correct classifications on the training set
		for(int i = 0; i < trainingInstances.size(); i++) {
			if(getClass(trainingInstances.get(i), root) == trainingInstances.get(i).get(attributes.size())) {
				correct++;
			}
		}

		// Calculate the training accuracy
		trainingAccuracy = Math.round(((double) correct / trainingInstances.size() * 100) * 10) / 10.0;

		// Print the training accuracy
		System.out.println("Accuracy on training set (" + trainingInstances.size() + " instances): " + trainingAccuracy + "%");

		correct = 0;

		// Calculate the amount of correct classifications on the testing set
		for(int i = 0; i < testingInstances.size(); i++) {
			if(getClass(testingInstances.get(i), root) == testingInstances.get(i).get(attributes.size())) {
				correct++;
			}
		}

		// Calculate the testing accuracy
		testingAccuracy = Math.round(((double) correct / testingInstances.size() * 100) * 10) / 10.0;

		// Print the testing accuracy
		System.out.println("Accuracy on test set (" + testingInstances.size() + " instances): " + testingAccuracy + "%");
	}

	/**
	* Get the classification of a given data point recursively
	*/
	private int getClass(ArrayList<Integer> instance, TreeNode node) {

		// If the node is a leaf, get the dominant class
		if(node.isLeaf()) {
			return node.getDominantClass();
		}

		// Go another level deeper on either class 0 or 1 on attribute that this node splits on
		if(instance.get(attributes.indexOf(node.getAttr())) == 0) {
			return getClass(instance, node.getZeroNode());
		}
		else {
			return getClass(instance, node.getOneNode());
		}
	}

	class TreeNode {

		// Amount of class 0 and 1 instances
		private int class0Count;
		private int class1Count;

		// Left and right node
		private TreeNode zero;
		private TreeNode one;

		// What attribute this node splits on
		private String splitAttr;

		// Data points in this node
		private ArrayList<ArrayList<Integer>> instances;

		// Remaining attribute to split on
		private ArrayList<String> remainingAttrs;

		public TreeNode() {
			class0Count = 0;
			class1Count = 0;

			zero = null;
			one = null;

			instances = new ArrayList<>();

			remainingAttrs = new ArrayList<>();
		}

		/**
		* Calculate the conditional entropy of this node
		*/
		public double getEntropy() {
			if(class0Count == 0 || class1Count == 0) {
				return 0;
			}

			double first = -((double) class0Count/(class0Count + class1Count)) * Math.log((double) class0Count/(class0Count + class1Count)) / Math.log(2);
			double second = -((double) class1Count/(class0Count + class1Count)) * Math.log((double) class1Count/(class0Count + class1Count)) / Math.log(2);

			return first + second;
		}

		/**
		* Get the class that wins this node
		*/
		public int getDominantClass() {
			if(class0Count > class1Count) {
				return 0;
			}
			else if(class1Count > class0Count) {
				return 1;
			}
			else {
				// If the counts are the same, return the class that appears the most in the tree 
				if(root.getClass0Count() > root.getClass1Count()) {
					return 0;
				}
				else {
					return 1;
				}
			}
		}

		/**
		* This node is a leaf if it doesn't have children
		*/
		public boolean isLeaf() {
			return zero == null && one == null ? true : false;
		}

		/**
		* Setters
		*/
		public void setRemainingAttrs(ArrayList<String> remainingAttrs) {
			this.remainingAttrs = remainingAttrs;
		}

		public void setZeroNode(TreeNode node) {
			zero = node;
		}

		public void setOneNode(TreeNode node) {
			one = node;
		}

		public void setClass0Count(int count) {
			class0Count = count;
		}

		public void setClass1Count(int count) {
			class1Count = count;
		}

		public void setAttr(String attr) {
			splitAttr = attr;
		}

		/**
		* Getters
		*/
		public ArrayList<String> getRemainingAttrs() {
			return remainingAttrs;
		}

		public TreeNode getZeroNode() {
			return zero;
		}

		public TreeNode getOneNode() {
			return one;
		}

		public int getClass0Count() {
			return class0Count;
		}

		public int getClass1Count() {
			return class1Count;
		}

		public String getAttr() {
			return splitAttr;
		}

		public int incrementClass0Count() {
			return ++class0Count;
		}

		public int incrementClass1Count() {
			return ++class1Count;
		}

		/**
		* Add data points to point storage
		*/
		public void addInstance(ArrayList<Integer> val) {
			instances.add(val);
		}

		public ArrayList<ArrayList<Integer>> getInstances() {
			return instances;
		}

		/**
		* Remove attribute from remaining attributes
		*/
		public void removeAttr(String attr) {
			remainingAttrs.remove(attr);
		}

		/**
		* Copy over given attribute list to the remaining attributes
		*/
		public void copyAttrs(ArrayList<String> attrs) {
			for(String attr : attrs) {
				remainingAttrs.add(attr);
			}
		}

		/**
		* Remove all remaining attributes remaining
		*/
		public void emptyAttrs() {
			remainingAttrs.clear();
		}

	}
}