import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
class stat { 
	int[] wordPositionCount = new int[] {0,0}; // # of unique word positions in +ve(0) and -ve(1) reviews 
	double[] ReviewCount = new double[] {0,0}; // at 0 -> positive review ; 1->negative review
	double totalReviews =0; 
	double posProb=0; // prior probability of positive review
	double negProb=0;//prior probability of negative review	
	// performance statistics
	float accuracy=0;
	float posPrecision=0;
	float posRecall=0;
	float posFmeasure=0;
	float negPrecision=0;
	float negRecall=0;
	float negFmeasure=0;
	int truePositive = 0,falsePositive = 0,trueNegative = 0,falseNegative = 0;
	
	/* the following methods return the specific performance measures for the class value passed
	 * class Type = 0 ---> Positive reviews
	 * class Type = 1 ---> Negative reviews
	 * */
	float Precision(int classType){
		float precision = 0;
		float temp=0;
		if(classType == 0) {
			temp = truePositive+falsePositive;
			posPrecision = (float)truePositive/temp;
			precision = posPrecision;
		}else {
			temp = trueNegative+falseNegative;
			negPrecision = (float)trueNegative/temp;
			precision = negPrecision;
		}
		return precision;
	}
	
	float Recall(int classType){
		float recall = 0;
		float temp=0;
		if(classType == 0) {			
			temp =truePositive+falseNegative;
			posRecall = (float)truePositive/temp;
			recall =posRecall;
		}else {
			temp =trueNegative+falsePositive;
			negRecall = (float)trueNegative/temp;
			recall =negRecall;
		}
		
		return recall;		
	}
	float fMeasure(int classType){
		float fMeasure= 0;
		float p,r;
		p = Precision(classType);
		r = Recall(classType);
		if(classType == 0) {
			posFmeasure = (2 * r * p) /(r+p);
			fMeasure = posFmeasure;
			
		}else {
			negFmeasure = (2 * r * p) /(r+p);
			fMeasure = negFmeasure;
		}
		
		return fMeasure;		
	}

}

public class Naive_Bayes_Classifier {
	static String vocabFile = "Dataset/vocab.txt";
	static String stopWordsFile = "Dataset/stopwords.txt";	
	static String positiveTrainFile = "Dataset/trainPos.txt";
	static String negativeTrainFile = "Dataset/trainNeg.txt";
	static String positiveTestFile = "Dataset/testPos.txt";
	static String negativeTestFile = "Dataset/testNeg.txt";
	static ArrayList<String> stopwords = new ArrayList<String>();
	
	
	private static stat LearnNaiveBayes(int type, Map<String, ArrayList<Double>> vMap) {
		/*  type:possible values[0-> basic naive bayes,1->removing stop words,2-> Binary NB]
		 *  vMap:vocabaulary,word count  and liklihood in every class
		 *  parses data and calulates liklihood and returns statistics
		 * */
		stat nb = new stat();
		int posReview = 0,negReview = 1;
		LoadVocab(vocabFile,vMap,type);
		ParseTrainingData(positiveTrainFile,posReview,nb,type,vMap);
		ParseTrainingData(negativeTrainFile,negReview,nb,type,vMap);
		calulateProbilities(nb,vMap);
		return nb;
	}
	private static void LoadStopWords(String filename) {
		// Loads the list of stop words from file into an arraylist - > stopWords
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line;			
			while ((line = br.readLine()) != null) {
				String word = line.trim();				
				stopwords.add(word);							
			}				
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	
	}	
	private static void calulateProbilities(stat nb, Map<String, ArrayList<Double>> vMap) {
		// calculates liklihood probabilities for every word in vocabulary
		nb.posProb =nb.ReviewCount[0]/nb.totalReviews;
		nb.negProb = nb.ReviewCount[1]/nb.totalReviews;
		int vocabSize = vMap.size();
		for (String key : vMap.keySet()) {
			ArrayList<Double> values = vMap.get(key);	
			double wordPosProb = ((values.get(0) + 1) / (nb.wordPositionCount[0]+vocabSize));
			double wordNegProb = ((values.get(1) + 1) / (nb.wordPositionCount[1]+vocabSize));
			values.set(2, wordPosProb);
			values.set(3, wordNegProb);
		}		
	}
	private static void ParseTrainingData(String filePath, int classification, stat nb, int flag, Map<String, ArrayList<Double>> vMap) {
		/*vbMap<Key,Values> : Key - word in the vocabulary file
		  values=arrayList of size 4
		  values-> index 0  -> number of time the word appeared in positive reviews
		  		   index 1  -> number of time the word appeared in negative reviews
		  		   index 2  -> probability (word/classification=positive)
		  	   	   index 3  -> probability (word/classification=negative)
		  	
		  	flag -> 0(basic naive bayes),1(ignore stopwords),2(binary NB)
		  	classification -> 0(psoitive),1(negative)
		  	this method -> reads file of specific class and counts occurance of each word
		  	  	   			based on the type(indicated by flag) 
		  */		
		try {			
			BufferedReader br = new BufferedReader(new FileReader(filePath));
			String line;
			while((line = br.readLine())!=null){
				String[] token =line.split(" +"); // update this reg expression accordingly
				nb.ReviewCount[classification]++;
				nb.totalReviews++;
				ArrayList<String> wordsSeen = new ArrayList<String>();
				int tokenCount = token.length;
				for(int count =0;count< tokenCount;count++) {
					String key = token[count].trim();
					if(key.startsWith("'")) {
						key = key.substring(1, key.length());
					}
					if(key.endsWith("'")) {
						key = key.substring(0, key.length()-1);
					}
														// add extra data clean up steps
					if(vMap.containsKey(key)) {
						if(flag == 2) {
							if(!wordsSeen.contains(key)) {
								wordsSeen.add(key);
								updateMap(key,classification,nb,vMap);	
							}
						}else {
						updateMap(key,classification,nb,vMap);	
					}					
					}
					else {
						int aposIndex = key.lastIndexOf('\'');
						if (aposIndex > 0) {
							String substring = key.substring(0, aposIndex);
							String mergedString = substring + key.substring(aposIndex+1);
							if(vMap.containsKey(substring)) {
								updateMap(substring,classification,nb,vMap);	
							}
							else if(vMap.containsKey(mergedString)) {							
								updateMap(mergedString,classification,nb,vMap);	

							}
							else {
							//	System.out.println(key);
							}
						}
					}
				}
				
			}
			br.close();				
		}catch(Exception e) {
			e.printStackTrace();
		}
	}


	private static void updateMap(String key, int classification, stat nb, Map<String, ArrayList<Double>> vMap) {
		//updates the respective count of the word in the respective vocab map 
		ArrayList<Double> values = vMap.get(key);
		double newValue = values.get(classification) + 1;
		values.set(classification, newValue);
		nb.wordPositionCount[classification]++;			
	}
	private static void LoadVocab(String Filename, Map<String, ArrayList<Double>> vMap, int type) {
		//Loading the vocab list of the data into the map and initializing values
		try {
			BufferedReader br = new BufferedReader(new FileReader(Filename));
			String line;			
			while ((line = br.readLine()) != null) {
				String key = line.trim();
				if(type == 1) { //the case where we ignore stopwords				{ 
					//don't add it to the vocab if it is one
					if(stopwords.contains(key))
						continue;
				}
				ArrayList<Double> valueList = new ArrayList<Double>();
				for(int j=0;j<4;j++) {
					valueList.add(0.0);
				}
				vMap.put(key,valueList);							
			}				
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static float test(String filePath,int expectedClass, int flag, stat nb, Map<String, ArrayList<Double>> vMap) {
		// read and classify each review for the respective class and update statistics
		int correctCount=0,wrongCount=0;
		try {			
			BufferedReader br = new BufferedReader(new FileReader(filePath));
			String line;
			while((line = br.readLine())!=null){
				int classifiedAs = classify(line,flag,nb,vMap);

				if(classifiedAs == expectedClass) {
					correctCount++;
				}
				else {
					wrongCount++;
				}
			}			
			if(expectedClass == 0) {
				nb.truePositive = correctCount;
				nb.falseNegative = wrongCount;
			}
			else {
				nb.trueNegative = correctCount;
				nb.falsePositive = wrongCount;
			}
			br.close();				
		}catch(Exception e) {
			e.printStackTrace();
		}
		return correctCount;
	}		
	private static int classify(String line, int flag, stat nb, Map<String, ArrayList<Double>> vMap) {		
		// classifies the give review as positive or negative
		double pPositive=0;// probablity of review being -ve
		double pNegative=0; // probability of review being +ve 
		double pPosTemp = 0,pNegTemp = 0; // to store intermeditae results of computation of liklihood
		ArrayList<String> wordsSeen = new ArrayList<String>(); // words from vocab seen in review before the current word
		String[] token =line.split(" +"); // update this reg expression accordingly			
		int tokenCount = token.length;
		for(int count =0;count< tokenCount;count++) {
			String key = token[count].trim();
			if(key.startsWith("'")) {
				key = key.substring(1, key.length());
			}
			if(key.endsWith("'")) {
				key = key.substring(0, key.length()-1);
			}
			// add extra data clean up steps
			if(vMap.containsKey(key)) {
					if(flag == 2) {
						if(!wordsSeen.contains(key)) {
							wordsSeen.add(key);
							ArrayList<Double> values = vMap.get(key);
							pPosTemp += Math.log(values.get(2));
							pNegTemp += Math.log(values.get(3));				
						}
					}else {
						ArrayList<Double> values = vMap.get(key);
						pPosTemp += Math.log(values.get(2));
						pNegTemp += Math.log(values.get(3));			
					}					
			}
			else {
				int aposIndex = key.lastIndexOf('\'');
				if (aposIndex > 0) {
					String substring = key.substring(0, aposIndex-1 );
					String mergedString = substring + key.substring(aposIndex+1);
					if(vMap.containsKey(substring)) {
						if(flag == 2) {
							if(!wordsSeen.contains(substring)) {
								wordsSeen.add(substring);
								ArrayList<Double> values = vMap.get(substring);
								pPosTemp += Math.log(values.get(2));
								pNegTemp += Math.log(values.get(3));				
							}
						}else {
							ArrayList<Double> values = vMap.get(substring);
							pPosTemp += Math.log(values.get(2));
							pNegTemp += Math.log(values.get(3));			
						}	
						
					}
					else if(vMap.containsKey(mergedString)) {	
						if(flag == 2) {
							if(!wordsSeen.contains(mergedString)) {
								wordsSeen.add(mergedString);
								ArrayList<Double> values = vMap.get(mergedString);
								pPosTemp += Math.log(values.get(2));
								pNegTemp += Math.log(values.get(3));				
							}
						}else {
							ArrayList<Double> values = vMap.get(mergedString);
							pPosTemp += Math.log(values.get(2));
							pNegTemp += Math.log(values.get(3));			
						}	

					}	
					
				}		
			}
		}		
		pPositive = Math.log(nb.posProb) + pPosTemp;
		pNegative = Math.log(nb.negProb) + pNegTemp;
		return( (pPositive > pNegative)?0:1);
	}
	private static void ClassifyNaiveBayes(int i, stat nb, Map<String, ArrayList<Double>> vMap) {
		// Testing the trained classifier
		int type = i;
		float correctPCount = test(positiveTestFile,0,type,nb,vMap);
		float correctNCount = test(negativeTestFile,1,type,nb,vMap);
		nb.accuracy = (float) ((correctPCount+correctNCount)/nb.totalReviews);
	}
	private static void OutputResults(stat nb1, stat nb2, stat nb3) {
		// outputs the results comparing all three classifiers
		//Analyse and output the results
				System.out.println("*******************************************************RESULTS*******************************************************");
				System.out.println(" ");
				System.out.println("------------------------------------------------------------------------------------------------------------------- ");
				System.out.println("Performance Measure\t|Naive Bayes \t\t|Ignored Stop Words\t\t|Binary NB \t|");
				System.out.println("------------------------------------------------------------------------------------------------------------------- ");
				System.out.println(" ");
				System.out.println("Accuracy\t\t|"+nb1.accuracy*100+"%\t\t|"+nb2.accuracy*100+"%\t\t\t|"+nb3.accuracy*100+"%\t|");
				System.out.println(" ");
				System.out.println("For Positive Reviews: ");
				System.out.println(" ");
				System.out.println("Precision\t\t|"+nb1.Precision(0)+"\t\t|"+nb2.Precision(0)+"\t\t\t|"+nb3.Precision(0)+"\t|");
				System.out.println(" ");
				System.out.println("Recall\t\t\t|"+nb1.Recall(0)+"\t\t|"+nb2.Recall(0)+"\t\t\t\t|"+nb3.Recall(0)+"\t|");
				System.out.println(" ");
				System.out.println("FMeasure\t\t|"+nb1.fMeasure(0)+"\t\t|"+nb2.fMeasure(0)+"\t\t\t|"+nb3.fMeasure(0)+"\t|");
				System.out.println(" ");
				System.out.println("For Negative Reviews:");
				System.out.println(" ");
				System.out.println("Precision\t\t|"+nb1.Precision(1)+"\t\t|"+nb2.Precision(1)+"\t\t\t|"+nb3.Precision(1)+"\t|");
				System.out.println(" ");
				System.out.println("Recall\t\t\t|"+nb1.Recall(1)+"\t\t|"+nb2.Recall(1)+"\t\t\t|"+nb3.Recall(1)+"\t|");
				System.out.println(" ");
				System.out.println("FMeasure\t\t|"+nb1.fMeasure(1)+"\t\t|"+nb2.fMeasure(1)+"\t\t\t|"+nb3.fMeasure(1)+"\t|");		
	}

	public static void main(String[] args) {
		Map<String,ArrayList<Double>> vocabMapNB = new LinkedHashMap<String,ArrayList<Double>>(); 
		Map<String,ArrayList<Double>> vocabMapSW = new LinkedHashMap<String,ArrayList<Double>>(); 
		Map<String,ArrayList<Double>> vocabMapBNB = new LinkedHashMap<String,ArrayList<Double>>();
		// Multinomial Naive Bayes
		stat nb1 = LearnNaiveBayes(0,vocabMapNB);
		ClassifyNaiveBayes(0,nb1,vocabMapNB);
		LoadStopWords(stopWordsFile);
		// Ignoring stop words
		stat nb2 = LearnNaiveBayes(1,vocabMapSW);// 1 -> stop words
		ClassifyNaiveBayes(1,nb2,vocabMapSW);
		// Binary Naive Bayes
		stat nb3 = LearnNaiveBayes(2,vocabMapBNB);
		ClassifyNaiveBayes(2,nb3,vocabMapBNB);
		OutputResults(nb1,nb2,nb3);
	}		
}
