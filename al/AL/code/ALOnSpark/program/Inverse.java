package al;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.SVD;

public class Inverse {
    private DenseMatrix A1;

    private int m;

    private int n;

    public Inverse(DenseMatrix AD){
        m = AD.numRows();
        n = AD.numColumns();

        A1 = AD.copy();

    }

    public DenseMatrix getInverse(){

        DenseMatrix I = Matrices.identity(n);
        DenseMatrix Ainv = I.copy();
        A1.solve(I, Ainv);

        return Ainv;
    }

    public DenseMatrix getMPInverse() throws Exception{
        SVD svd= new SVD(m,n);
        svd.factor(A1);
        DenseMatrix U = svd.getU();
        DenseMatrix Vt = svd.getVt();
        double []s = svd.getS();
        int sn = s.length;
        for (int i = 0; i < sn; i++) {
            s[i] = Math.sqrt(s[i]);
        }

        DenseMatrix S1 = (DenseMatrix) Matrices.random(m, sn);
        S1.zero();
        DenseMatrix S2 = (DenseMatrix) Matrices.random(sn, n);
        S2.zero();
        for (int i = 0; i < s.length; i++) {
            S1.set(i, i, s[i]);
            S2.set(i, i, s[i]);
        }

        DenseMatrix C = new DenseMatrix(m,sn);
        U.mult(S1, C);
        DenseMatrix D = new DenseMatrix(sn,n);
        S2.mult(Vt,D);

        DenseMatrix DD = new DenseMatrix(sn,sn);
        DenseMatrix DT = new DenseMatrix(n,sn);
        D.transpose(DT);
        D.mult(DT, DD);
        Inverse inv = new Inverse(DD);
        DenseMatrix invDD = inv.getInverse();

        DenseMatrix DDD = new DenseMatrix(n,sn);
        DT.mult(invDD, DDD);

        DenseMatrix CC = new DenseMatrix(sn,sn);
        DenseMatrix CT = new DenseMatrix(sn,m);
        C.transpose(CT);

        CT.mult(C, CC);
        Inverse inv2 = new Inverse(CC);
        DenseMatrix invCC = inv2.getInverse();

        DenseMatrix CCC = new DenseMatrix(sn,m);
        invCC.mult(CT, CCC);

        DenseMatrix Ainv = new DenseMatrix(n,m);
        DDD.mult(CCC, Ainv);
        return Ainv;
    }
    public DenseMatrix getMPInverse(double lumda) throws Exception{
        DenseMatrix At = new DenseMatrix(n, m);
        A1.transpose(At);
        DenseMatrix AtA = new DenseMatrix(n ,n);
        At.mult(A1,AtA);

        DenseMatrix I = Matrices.identity(n);
        AtA.add(lumda, I);
        DenseMatrix AtAinv = I.copy();
        AtA.solve(I, AtAinv);

        DenseMatrix Ainv = new DenseMatrix(n,m);
        AtAinv.mult(At, Ainv);
        return Ainv;
    }
    public DenseMatrix checkCD() throws Exception{
        SVD svd= new SVD(m,n);		//U*S*Vt=A;
        svd.factor(A1);
        DenseMatrix U = svd.getU();		//m*m
        DenseMatrix Vt = svd.getVt();	//n*n
        double []s = svd.getS();
        int sn = s.length;
        for (int i = 0; i < s.length; i++) {
            s[i] = Math.sqrt(s[i]);
        }
        DenseMatrix S1 = (DenseMatrix) Matrices.random(m, sn);
        S1.zero();
        DenseMatrix S2 = (DenseMatrix) Matrices.random(sn, n);
        S2.zero();
        for (int i = 0; i < s.length; i++) {
            S1.set(i, i, s[i]);
            S2.set(i, i, s[i]);
        }

        DenseMatrix C = new DenseMatrix(m,sn);
        U.mult(S1, C);
        DenseMatrix D = new DenseMatrix(sn,n);
        S2.mult(Vt,D);

        DenseMatrix CD = new DenseMatrix(m,n);
        C.mult(D, CD);

        return CD;
    }
}
