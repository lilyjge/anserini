package io.anserini.fusion;

import java.util.ArrayList;

import org.apache.commons.lang3.ArrayUtils;

public class Hits{
    public String[] query;
    public String[] docid;
    public double[] score;
    public int[] rank;
    public Hits(int size){
      this.query = new String[size];
      this.docid = new String[size];
      this.rank = new int[size];
      this.score = new double[size];
    }
    public Hits() {
      this.docid = null;
      this.query = null;
      this.rank = null;
      this.score = null;
    }
    public Hits(ArrayList<String> queries, ArrayList<String> docids, ArrayList<Double> scores, ArrayList<Integer> ranks){
      this.docid = docids.toArray(new String[0]);
      this.query = queries.toArray(new String[0]);
      this.rank = ArrayUtils.toPrimitive(ranks.toArray(new Integer[0]));
      this.score = ArrayUtils.toPrimitive(scores.toArray(new Double[0]));
    }

    // public String toString(Hit hit){
    //   return hit.query + " " + hit.docid + " " + hit.rank + " " + hit.score;
    // }
  }