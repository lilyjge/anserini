/*
 * Anserini: A Lucene toolkit for reproducible information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.fusion;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.kohsuke.args4j.Option;

import io.anserini.search.ScoredDocs;

import java.time.Duration;
import java.time.Instant;
/**
 * Main logic class for Fusion
 */
public class RunsFuser {
  private final Args args;

  private static final String METHOD_RRF = "rrf";
  private static final String APPEND = "append";
  private static final String METHOD_INTERPOLATION = "interpolation";
  private static final String METHOD_AVERAGE = "average";
  private static final String METHOD_NORMALIZE = "normalize";

  public static class Args {
    @Option(name = "-output", metaVar = "[output]", required = true, usage = "Path to save the output")
    public String output;

    @Option(name = "-runtag", metaVar = "[runtag]", required = false, usage = "Run tag for the fusion")
    public String runtag = "anserini.fusion";

    @Option(name = "-method", metaVar = "[method]", required = false, usage = "Specify fusion method")
    public String method = "rrf";

    @Option(name = "-rrf_k", metaVar = "[number]", required = false, usage = "Parameter k needed for reciprocal rank fusion.")
    public int rrf_k = 60;

    @Option(name = "-alpha", metaVar = "[value]", required = false, usage = "Alpha value used for interpolation.")
    public double alpha = 0.5;

    @Option(name = "-k", metaVar = "[number]", required = false, usage = "number of documents to output for topic")
    public int k = 1000;

    @Option(name = "-depth", metaVar = "[number]", required = false, usage = "Pool depth per topic.")
    public int depth = 1000;
  }

  public RunsFuser(Args args) {
    this.args = args;
  }

  public record QueryAndDoc(String query, String doc) {}
  
  /**
   * Perform fusion by averaging on a list of ScoredDocs objects.
   *
   * @param runs List of ScoredDocs objects.
   * @param depth Maximum number of results from each input run to consider. Set to Integer.MAX_VALUE by default, which indicates that the complete list of results is considered.
   * @param k Length of final results list. Set to Integer.MAX_VALUE by default, which indicates that the union of all input documents are ranked.
   * @return Output ScoredDocs that combines input runs via averaging.
   */
  public static Hits average(List<Hits> runs, int depth, int k) {
    for (Hits run : runs) {
      HitsFuser.rescore(RescoreMethod.SCALE, 0, (1/(double)runs.size()), run);;
    }

    return HitsFuser.merge(runs, depth, k);
  }

  /**
   * Perform fusion by normalizing scores and taking the average. 
   *
   * @param runs List of ScoredDocs objects.
   * @param depth Maximum number of results from each input run to consider. Set to Integer.MAX_VALUE by default, which indicates that the complete list of results is considered.
   * @param k Length of final results list. Set to Integer.MAX_VALUE by default, which indicates that the union of all input documents are ranked.
   * @return Output ScoredDocs that combines input runs via reciprocal rank fusion.
   */
  public static Hits normalize(List<Hits> runs, int depth, int k) {
    for (Hits run : runs) {
      HitsFuser.rescore(RescoreMethod.NORMALIZE, 0, 0, run);
    }

    return average(runs, depth, k);
  }

  /**
   * Perform fusion by interpolation on a list of exactly two ScoredDocs objects.
   * new_score = first_run_score * alpha + (1 - alpha) * second_run_score.
   *
   * @param runs List of ScoredDocs objects. Exactly two runs.
   * @param alpha Parameter alpha will be applied on the first run and (1 - alpha) will be applied on the second run.
   * @param depth Maximum number of results from each input run to consider. Set to Integer.MAX_VALUE by default, which indicates that the complete list of results is considered.
   * @param k Length of final results list. Set to Integer.MAX_VALUE by default, which indicates that the union of all input documents are ranked.
   * @return Output ScoredDocs that combines input runs via interpolation.
   */  
  public static Hits interpolation(List<Hits> runs, double alpha, int depth, int k) {
    // Ensure exactly 2 runs are provided, as interpolation requires 2 runs
    if (runs.size() != 2) {
      throw new IllegalArgumentException("Interpolation requires exactly 2 runs");
    }

    HitsFuser.rescore(RescoreMethod.SCALE, 0, alpha, runs.get(0));
    HitsFuser.rescore(RescoreMethod.SCALE, 0, 1 - alpha, runs.get(1));

    return HitsFuser.merge(runs, depth, k);
  }

  public static Hits rrf(List<Hits> runs, int rrf_k, int depth, int k) {
    // Instant start = Instant.now();
    for (Hits run : runs) {
      HitsFuser.rescore(RescoreMethod.RRF, rrf_k, 0, run);
    }
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Rescoring runs: "+ timeElapsed.toSeconds() +" seconds");
    return HitsFuser.merge(runs, depth, k);
  }

  // public record Hit(Document query, String docid, double score, int rank){}

  public static Hits rrf3(List<Hits> runs, int rrf_k, int depth, int topN){
    // Instant start = Instant.now();
    int newLength = 0; // find length for appended array
    for (Hits scoredDocs : runs){
      newLength += scoredDocs.docid.length;
    }
    Hits scores = new Hits(newLength);
    int index = 0;
    for (Hits scoredDocs : runs){
      for(int i = 0; i < scoredDocs.docid.length; i++){
        scores.docid[index + i] = scoredDocs.docid[i];
        scores.query[index + i] = scoredDocs.query[i];
        scores.rank[index + i] = scoredDocs.rank[i];
        scores.score[index + i] = 1 / ((double)scoredDocs.rank[i] + rrf_k);
        // System.out.println(scores[index + i].score);
        // scores[index + 1] = new Hit(scoredDocs.lucene_documents[i], scoredDocs.docids[i], 1 / (double)(scoredDocs.lucene_docids[i] + (float)rrf_k), 0);
      }
      index += scoredDocs.docid.length;
    }
    // System.out.println("appended " + index);
    HitsFuser.sortHits(scores, true);
    // System.out.println("sorted");
    int write = 0;
    String curQuery = scores.query[0];
    String curDoc = scores.docid[0];
    for(int i = 1; i < scores.docid.length; i++){
      if(scores.docid[i].equals(curDoc) && scores.query[i].equals(curQuery)){
        scores.score[write] += scores.score[i];
      }
      else{
        write++;
        scores.docid[write] = scores.docid[i];
        scores.query[write] = scores.query[i];
        scores.score[write] = scores.score[i];
        scores.rank[write] = scores.rank[i];
        curQuery = scores.query[write];
        curDoc = scores.docid[write];
      }
    }
    write++;
    Hits newScores = new Hits(write);
    for(int i = 0; i < write; i++){
      newScores.docid[i] = scores.docid[i];
      newScores.query[i] = scores.query[i];
      newScores.score[i] = scores.score[i];
      newScores.rank[i] = scores.rank[i];
    }
    // System.out.println(write);
    HitsFuser.sortHits(newScores, false);
    // System.out.println("sorted");
    write = 0;
    int rank = 0;
    curQuery = "";
    for(int i = 0; i < newScores.docid.length; i++){
      if(curQuery.equals(newScores.query[i]) && rank > topN){
        continue;
      }
      else if(curQuery.equals(newScores.query[i])){
        rank++;
      }
      else{
        rank = 1;
        curQuery = newScores.query[i];
      }
      newScores.query[write] = newScores.query[i];
      newScores.rank[write] = rank;
      newScores.docid[write] = newScores.docid[i];
      newScores.score[write] = newScores.score[i];
      write++;
    }
    // System.out.println(write);
    scores = new Hits(write);
    for(int i = 0; i < write; i++){
      scores.docid[i] = newScores.docid[i];
      scores.query[i] = newScores.query[i];
      scores.score[i] = newScores.score[i];
      scores.rank[i] = newScores.rank[i];
    }
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Total: "+ timeElapsed.toSeconds() +" seconds");
    return scores;
  }

  public static ScoredDocs rrf2(List<ScoredDocs> runs, int rrf_k, int depth, int topN){
    // Instant start = Instant.now();
    // HashSet<String> queries = new HashSet<String>(); // to calculate arr len based on topN
    // HashSet<String> queryDocs = new HashSet<String>(); // max possible arr len

    int newLength = 0; // find length for appended array
    for (ScoredDocs scoredDocs : runs){
      newLength += scoredDocs.lucene_documents.length;
    }

    ScoredDocs appended = new ScoredDocs();
    appended.lucene_documents = new Document[newLength];
    appended.docids = new String[newLength];
    appended.scores = new float[newLength];

    // appending the arrays
    int index = 0;
    for (ScoredDocs scoredDocs : runs){
      for(int i = 0; i < scoredDocs.lucene_documents.length; i++){
        appended.lucene_documents[index + i] = scoredDocs.lucene_documents[i];
        appended.docids[index + i] = scoredDocs.docids[i];
        appended.scores[index + i] = 1 / (float)(scoredDocs.lucene_docids[i] + (float)rrf_k);
        // String query = scoredDocs.lucene_documents[i].get("TOPIC"); 
        // queryDocs.add(query + " " + scoredDocs.docids[i]);
        // queries.add(query);
      }
      index += scoredDocs.lucene_documents.length;
    }
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Appending runs: "+ timeElapsed.toSeconds() +" seconds");
    // start = end;
    // System.out.println(queries);

    // sorting by query then docid
    ScoredDocsFuser.sortScoredDocs(appended, false, 0);
    ScoredDocs summed = new ScoredDocs();
    // newLength = queryDocs.size();
    // summed.lucene_documents = new Document[newLength];
    // summed.docids = new String[newLength];
    // summed.scores = new float[newLength];
    List<Document> lucene_documents = new ArrayList<>(newLength); // topic
    List<String> docids = new ArrayList<>(newLength); // docid
    List<Float> scores = new ArrayList<>(newLength); // score

    String curQuery = "", curDoc = "";
    index = 0;
    // System.out.println(newLength); // length after summing same query, samd docid scores
    // summing scores for same query, docids
    for(int i = 0; i < appended.lucene_documents.length; i++){
      curQuery = appended.lucene_documents[i].get("TOPIC");
      curDoc = appended.docids[i];
      float curScore = 0;
      while(i < appended.lucene_documents.length && curQuery.equals(appended.lucene_documents[i].get("TOPIC")) && curDoc.equals(appended.docids[i])){
        curScore += appended.scores[i];
        i++;
      }
      i--;
      // summed.lucene_documents[index] = appended.lucene_documents[i];
      // summed.docids[index] = curDoc;
      // summed.scores[index] = curScore;
      lucene_documents.add(appended.lucene_documents[i]);
      docids.add(curDoc);
      scores.add(curScore);
      index++;
    }
    // end = Instant.now();
    // timeElapsed = Duration.between(start, end);
    // System.out.println("Summing scores: "+ timeElapsed.toSeconds() +" seconds");
    // start = end;
  
    summed.lucene_documents = lucene_documents.toArray(new Document[0]);
    summed.docids = docids.toArray(new String[0]);
    summed.scores = ArrayUtils.toPrimitive(scores.toArray(new Float[scores.size()]), Float.NaN);

    int overflow = ScoredDocsFuser.sortScoredDocs(summed, true, topN);
    newLength -= overflow;
    Document[] lucene_documents1 = new Document[newLength];
    int[] lucene_docids1 = new int[newLength];
    String[] docids1 = new String[newLength];
    float[] scores1 = new float[newLength];
    index = 0;
    for(int i = 0; i < summed.lucene_documents.length; i++){
      if(summed.lucene_docids[i] > topN){
        continue;
      }
      lucene_documents1[index] = summed.lucene_documents[i];
      lucene_docids1[index] = summed.lucene_docids[i];
      docids1[index] = summed.docids[i];
      scores1[index] = summed.scores[i];
      index++;
    }
    summed.lucene_docids = lucene_docids1;
    summed.lucene_documents = lucene_documents1;
    summed.scores = scores1;
    summed.docids = docids1;
    // end = Instant.now();
    // timeElapsed = Duration.between(start, end);
    // System.out.println("Sorting by score and limit top n: "+ timeElapsed.toSeconds() +" seconds");
    return summed;
  }

  /**
   * Process the fusion of ScoredDocs objects based on the specified method.
   *
   * @param runs List of ScoredDocs objects to be fused.
   * @throws IOException If an I/O error occurs while saving the output.
   */
  public void fuse(List<Hits> runs) throws IOException {
    Instant start = Instant.now();
    Hits fusedRun;

    // Select fusion method
    switch (args.method.toLowerCase()) {
      case METHOD_RRF:
        fusedRun = rrf(runs, args.rrf_k, args.depth, args.k);
        break;
      case APPEND:
        fusedRun = rrf3(runs, args.rrf_k, args.depth, args.k);
        break;
      case METHOD_INTERPOLATION:
        fusedRun = interpolation(runs, args.alpha, args.depth, args.k);
        break;
      case METHOD_AVERAGE:
        fusedRun = average(runs, args.depth, args.k);
        break;
      case METHOD_NORMALIZE:
        fusedRun = normalize(runs, args.depth, args.k);
        break;
      default:
        throw new IllegalArgumentException("Unknown fusion method: " + args.method + 
            ". Supported methods are: average, rrf, interpolation.");
    }

    Path outputPath = Paths.get(args.output);
    HitsFuser.saveToTxt(outputPath, args.runtag,  fusedRun);
    Instant end = Instant.now();
    Duration timeElapsed = Duration.between(start, end);
    System.out.println("Total: "+ timeElapsed.toSeconds() +" seconds");
  }
}
