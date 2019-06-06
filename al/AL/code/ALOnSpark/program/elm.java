package al;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import java.io.Serializable;
import java.io.IOException;
//public class elm implements Writable{
public class elm implements Serializable{
    private int numTrainData;
    private int numTestData;
    private float TrainingTime;
    private float TestingTime;
    private double TrainingAccuracy, TestingAccuracy;
    private int Elm_Type;
    private int NumberofHiddenNeurons;
    private int NumberofOutputNeurons;
    private int NumberofInputNeurons;
    private String func;
    private int []label;
    private DenseMatrix BiasofHiddenNeurons;
    private DenseMatrix OutputWeight;
    private DenseMatrix testP;
    private DenseMatrix testT;
    private DenseMatrix Y;
    private DenseMatrix T;
    private DenseMatrix InputWeight;
    private DenseMatrix train_set;
    private DenseMatrix test_set;

    private DenseMatrix OutMatrix;

    public elm(int elm_type, int numberofHiddenNeurons, String ActivationFunction){
        Elm_Type = elm_type;
        NumberofHiddenNeurons = numberofHiddenNeurons;
        func = ActivationFunction;
        TrainingTime = 0;
        TestingTime = 0;
        TrainingAccuracy= 0;
        TestingAccuracy = 0;
        NumberofOutputNeurons = 1;

    }
    public elm(){
    }
    public void train(double [][]traindata,int NumberofOutputNeurons) throws Exception{
        this.NumberofOutputNeurons=NumberofOutputNeurons;
        //classification require a the number of class

        train_set = new DenseMatrix(traindata);
        int m = train_set.numRows();
        if(Elm_Type == 1){
            double maxtag = traindata[0][0];
            for (int i = 0; i < m; i++) {
                if(traindata[i][0] > maxtag)
                    maxtag = traindata[i][0];
            }
            NumberofOutputNeurons = (int)maxtag+1;
        }
        train();
    }
    private void train() throws Exception{

        numTrainData = train_set.numRows();
        NumberofInputNeurons = train_set.numColumns() - 1;
        InputWeight = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);

        DenseMatrix transT = new DenseMatrix(numTrainData, 1);
        DenseMatrix transP = new DenseMatrix(numTrainData, NumberofInputNeurons);
        for (int i = 0; i < numTrainData; i++) {
            transT.set(i, 0, train_set.get(i, 0));
            for (int j = 1; j <= NumberofInputNeurons; j++)
                transP.set(i, j-1, train_set.get(i, j));
        }
        T = new DenseMatrix(1,numTrainData);
        DenseMatrix P = new DenseMatrix(NumberofInputNeurons,numTrainData);
        transT.transpose(T);
        transP.transpose(P);

        if(Elm_Type != 0)
        {
            label = new int[NumberofOutputNeurons];
            for (int i = 0; i < NumberofOutputNeurons; i++) {
                label[i] = i;
            }
            DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,numTrainData);
            tempT.zero();
            for (int i = 0; i < numTrainData; i++){
                int j = 0;
                for (j = 0; j < NumberofOutputNeurons; j++){
                    if (label[j] == T.get(0, i))
                        break;
                }
                tempT.set(j, i, 1);
            }

            T = new DenseMatrix(NumberofOutputNeurons,numTrainData);
            for (int i = 0; i < NumberofOutputNeurons; i++){
                for (int j = 0; j < numTrainData; j++)
                    T.set(i, j, tempT.get(i, j)*2-1);
            }
            transT = new DenseMatrix(numTrainData,NumberofOutputNeurons);
            T.transpose(transT);

        }
        long start_time_train = System.currentTimeMillis();
        BiasofHiddenNeurons = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, 1);
        DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
        InputWeight.mult(P, tempH);

        DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons, numTrainData);

        for (int j = 0; j < numTrainData; j++) {
            for (int i = 0; i < NumberofHiddenNeurons; i++) {
                BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }
        tempH.add(BiasMatrix);
        DenseMatrix H = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
        if(func.startsWith("sig")){
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTrainData; i++) {
                    double temp = tempH.get(j, i);
                    temp = 1.0f/ (1 + Math.exp(-temp));
                    H.set(j, i, temp);
                }
            }
        }
        else if(func.startsWith("sin")) {
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTrainData; i++) {
                    double temp = tempH.get(j, i);
                    temp = Math.sin(temp);
                    H.set(j, i, temp);
                }
            }
        }

        DenseMatrix Ht = new DenseMatrix(numTrainData,NumberofHiddenNeurons);
        H.transpose(Ht);
        Inverse invers = new Inverse(Ht);
        DenseMatrix pinvHt = invers.getMPInverse();
        OutputWeight = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
        pinvHt.mult(transT, OutputWeight);

        long end_time_train = System.currentTimeMillis();
        TrainingTime = (end_time_train - start_time_train)*1.0f/1000;

        DenseMatrix Yt = new DenseMatrix(numTrainData,NumberofOutputNeurons);
        Ht.mult(OutputWeight,Yt);
        Y = new DenseMatrix(NumberofOutputNeurons,numTrainData);
        Yt.transpose(Y);

        if(Elm_Type == 0){
            double MSE = 0;
            for (int i = 0; i < numTrainData; i++) {
                MSE += (Yt.get(i, 0) - transT.get(i, 0))*(Yt.get(i, 0) - transT.get(i, 0));
            }
            TrainingAccuracy = Math.sqrt(MSE/numTrainData);
        }
        else if(Elm_Type == 1){
            float MissClassificationRate_Training=0;
            for (int i = 0; i < numTrainData; i++) {
                double maxtag1 = Y.get(0, i);
                int tag1 = 0;
                double maxtag2 = T.get(0, i);
                int tag2 = 0;
                for (int j = 1; j < NumberofOutputNeurons; j++) {
                    if(Y.get(j, i) > maxtag1){
                        maxtag1 = Y.get(j, i);
                        tag1 = j;
                    }
                    if(T.get(j, i) > maxtag2){
                        maxtag2 = T.get(j, i);
                        tag2 = j;
                    }
                }
                if(tag1 != tag2)
                    MissClassificationRate_Training ++;
            }
            TrainingAccuracy = 1 - MissClassificationRate_Training*1.0f/numTrainData;
        }
    }
    public void test(DenseMatrix  d)throws IOException{

        test_set=d.copy();
        numTestData = test_set.numRows();   
        DenseMatrix ttestT = new DenseMatrix(numTestData, 1);	
        DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);	
        for (int i = 0; i < numTestData; i++) {			
            ttestT.set(i, 0, test_set.get(i, 0));
            for (int j = 1; j <= NumberofInputNeurons; j++)
                ttestP.set(i, j-1, test_set.get(i, j));
        }

        testT = new DenseMatrix(1,numTestData);			
        testP = new DenseMatrix(NumberofInputNeurons,numTestData);		
        ttestT.transpose(testT);
        ttestP.transpose(testP);

        long start_time_test = System.currentTimeMillis();		
        DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);	
        InputWeight.mult(testP, tempH_test);	
        DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);	

        for (int j = 0; j < numTestData; j++) {
            for (int i = 0; i < NumberofHiddenNeurons; i++) {
                BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }

        tempH_test.add(BiasMatrix2);
        DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);

        if(func.startsWith("sig")){
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTestData; i++) {
                    double temp = tempH_test.get(j, i);
                    temp = 1.0f/ (1 + Math.exp(-temp));
                    H_test.set(j, i, temp);
                }
            }
        }
        else if(func.startsWith("sin")){
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTestData; i++) {
                    double temp = tempH_test.get(j, i);
                    temp = Math.sin(temp);
                    H_test.set(j, i, temp);
                }
            }
        }
        else if(func.startsWith("hardlim")){

        }
        else if(func.startsWith("tribas")){

        }
        else if(func.startsWith("radbas")){

        }

        DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
        H_test.transpose(transH_test);
        DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);
        transH_test.mult(OutputWeight,Yout);

        //*****shenchu***
        OutMatrix=new DenseMatrix(numTestData, NumberofOutputNeurons);
        OutMatrix=Yout.copy();

        DenseMatrix testY = new DenseMatrix(NumberofOutputNeurons,numTestData);
        Yout.transpose(testY);

        long end_time_test = System.currentTimeMillis();
        TestingTime = (end_time_test - start_time_test)*1.0f/1000;
        //writefile(Yout, "C:\\Users\\shen\\Desktop\\write1.txt");
        if(Elm_Type == 0){
            double MSE = 0;
            for (int i = 0; i < numTestData; i++) {
                MSE += (Yout.get(i, 0) - testT.get(0,i))*(Yout.get(i, 0) - testT.get(0,i));
            }
            TestingAccuracy = Math.sqrt(MSE/numTestData);
        }


        else if(Elm_Type == 1){

            DenseMatrix temptestT = new DenseMatrix(NumberofOutputNeurons,numTestData);
            for (int i = 0; i < numTestData; i++){
                int j = 0;
                for (j = 0; j < NumberofOutputNeurons; j++){
                    if (label[j] == testT.get(0, i))
                        break;
                }
                temptestT.set(j, i, 1);
            }

            testT = new DenseMatrix(NumberofOutputNeurons,numTestData);	
            for (int i = 0; i < NumberofOutputNeurons; i++){
                for (int j = 0; j < numTestData; j++)
                    testT.set(i, j, temptestT.get(i, j)*2-1);
            }

            float MissClassificationRate_Testing=0;

            for (int i = 0; i < numTestData; i++) {
                double maxtag1 = testY.get(0, i);
                int tag1 = 0;
                double maxtag2 = testT.get(0, i);
                int tag2 = 0;
                for (int j = 1; j < NumberofOutputNeurons; j++) {
                    if(testY.get(j, i) > maxtag1){
                        maxtag1 = testY.get(j, i);
                        tag1 = j;
                    }
                    if(testT.get(j, i) > maxtag2){
                        maxtag2 = testT.get(j, i);
                        tag2 = j;
                    }
                }
                if(tag1 != tag2)
                    MissClassificationRate_Testing ++;
            }
            TestingAccuracy = 1 - MissClassificationRate_Testing*1.0f/numTestData;

        }
    }
    public Double testHidenLabelOut(double[] inpt,int NumberofOutputNeurons){
        this.NumberofOutputNeurons=NumberofOutputNeurons;
        test_set = new DenseMatrix(new DenseVector(inpt));
        return testHidenLabelOut();
    }
    private double testHidenLabelOut(){
        numTestData = test_set.numColumns();
        NumberofInputNeurons = test_set.numRows()-1;

        DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);
        for(int j=1;j<=NumberofInputNeurons;j++){
            ttestP.set(0,j-1, test_set.get(j,0));
        }

        testP = new DenseMatrix(NumberofInputNeurons,numTestData);

        ttestP.transpose(testP);
        DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
        InputWeight.mult(testP, tempH_test);
        DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);
        for (int j = 0; j < numTestData; j++) {
            for (int i = 0; i < NumberofHiddenNeurons; i++) {
                BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
            }
        }
        tempH_test.add(BiasMatrix2);
        DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
        if(func.startsWith("sig")){
            for (int j = 0; j < NumberofHiddenNeurons; j++) {
                for (int i = 0; i < numTestData; i++) {
                    double temp = tempH_test.get(j, i);
                    temp = 1.0f/ (1 + Math.exp(-temp));
                    H_test.set(j, i, temp);
                }
            }
        }
        DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
        H_test.transpose(transH_test);
        DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);
        transH_test.mult(OutputWeight,Yout);
        double[] RYout=new double[NumberofOutputNeurons+1];
        RYout[NumberofOutputNeurons]=0;
        for(int i=0;i<NumberofOutputNeurons;i++){
            RYout[i]=Math.exp(Yout.get(0,i));
            RYout[NumberofOutputNeurons]+=RYout[i];
        }
        for(int i=0;i<NumberofOutputNeurons;i++){
            RYout[i]=RYout[i]/RYout[NumberofOutputNeurons];
        }
        double SoftMaxYout=0.0;
        for(int i=0;i<NumberofOutputNeurons;i++){
            SoftMaxYout+=RYout[i]*Math.log(RYout[i]);
        }
        return -SoftMaxYout;
    }
    public DenseMatrix getInputWeight() {	
        return InputWeight;
    }
    public DenseMatrix getBiasofHiddenNeurons() {	
        return BiasofHiddenNeurons;
    }
    public DenseMatrix getOutputWeight() {	
        return OutputWeight;
    }
    public double getTrainingAccuracy() {	
        return TrainingAccuracy;
    }
    public float getTrainingTime() {  
        return TrainingTime;
    }
    public int getNumberofOutputNeurons() {	
        return NumberofOutputNeurons;
    }
}
