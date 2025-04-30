package io.anserini.fusion;

public class Hit{
    public String query;
    public String docid;
    public double score;
    public int rank;
    public Hit(String query, String docid, int rank, double score){
      this.query = query;
      this.docid = docid;
      this.rank = rank;
      this.score = score;
    }
    public Hit() {
      this.docid = "";
      this.query = "";
      this.rank = -1;
      this.score = -1;
    }
    public Hit(Hit other){
      this.docid = other.docid;
      this.query = other.query;
      this.rank = other.rank;
      this.score = other.score;
    }

    public String toString(Hit hit){
      return hit.query + " " + hit.docid + " " + hit.rank + " " + hit.score;
    }
  }