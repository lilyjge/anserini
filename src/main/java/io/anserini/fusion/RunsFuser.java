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
  private static final String LUCENE = "lucenerrf";
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
  public static ScoredDocs average(List<ScoredDocs> runs, int depth, int k) {
    for (ScoredDocs run : runs) {
      ScoredDocsFuser.rescore(RescoreMethod.SCALE, 0, (1/(double)runs.size()), run);
    }

    return ScoredDocsFuser.merge(runs, depth, k);
  }

  /**
   * Perform reciprocal rank fusion on a list of TrecRun objects. Implementation follows Cormack et al.
   * (SIGIR 2009) paper titled "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods."
   *
   * @param runs List of ScoredDocs objects.
   * @param rrf_k Parameter to avoid vanishing importance of lower-ranked documents. Note that this is different from the *k* in top *k* retrieval; set to 60 by default, per Cormack et al.
   * @param depth Maximum number of results from each input run to consider. Set to Integer.MAX_VALUE by default, which indicates that the complete list of results is considered.
   * @param topN Length of final results list. Set to Integer.MAX_VALUE by default, which indicates that the union of all input documents are ranked.
   * @return Output ScoredDocs that combines input runs via reciprocal rank fusion.
   */
  public static ScoredDocs reciprocalRankFusion(List<ScoredDocs> runs, int rrf_k, int depth, int topN) {
    // Compute the rrf score as a double to reduce accuracy loss due to floating-point arithmetic.
    // Instant start = Instant.now();
    Map<QueryAndDoc, Double> rrfScore = new HashMap<>();
    HashSet<String> queries = new HashSet<String>();
    // HashSet<String> queryDocs = new HashSet<String>();
    for (ScoredDocs topDoc : runs) {
      // A document is a hit globally if it is a hit for any of the top docs, so we compute the
      // total hit count as the max total hit count.
      for (int i = 0; i < topDoc.lucene_documents.length; ++i) {
        int rank = topDoc.lucene_docids[i];
        double rrfScoreContribution = 1d / Math.addExact(rrf_k, rank);
        String query = topDoc.lucene_documents[i].get(ScoredDocsFuser.TOPIC);
        rrfScore.compute(
            new QueryAndDoc(query, topDoc.docids[i]),
            (qd, score) -> (score == null ? 0 : score) + rrfScoreContribution);
        queries.add(query);
      }
    }
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Getting map of qd and scores: "+ timeElapsed.toSeconds() +" seconds");

    return merge(rrfScore, Math.min(topN * queries.size(), rrfScore.size()), topN);
  }

  /**
   * Perform fusion by normalizing scores and taking the average. 
   *
   * @param runs List of ScoredDocs objects.
   * @param depth Maximum number of results from each input run to consider. Set to Integer.MAX_VALUE by default, which indicates that the complete list of results is considered.
   * @param k Length of final results list. Set to Integer.MAX_VALUE by default, which indicates that the union of all input documents are ranked.
   * @return Output ScoredDocs that combines input runs via reciprocal rank fusion.
   */
  public static ScoredDocs normalize(List<ScoredDocs> runs, int depth, int k) {
    for (ScoredDocs run : runs) {
      ScoredDocsFuser.rescore(RescoreMethod.NORMALIZE, 0, 0, run);
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
  public static ScoredDocs interpolation(List<ScoredDocs> runs, double alpha, int depth, int k) {
    // Ensure exactly 2 runs are provided, as interpolation requires 2 runs
    if (runs.size() != 2) {
      throw new IllegalArgumentException("Interpolation requires exactly 2 runs");
    }

    ScoredDocsFuser.rescore(RescoreMethod.SCALE, 0, alpha, runs.get(0));
    ScoredDocsFuser.rescore(RescoreMethod.SCALE, 0, 1 - alpha, runs.get(1));

    return ScoredDocsFuser.merge(runs, depth, k);
  }

  /**
   * Merges multiple ScoredDocs instances into a single ScoredDocs instance.
   * The merged ScoredDocs will contain the top documents for each topic, with scores summed across the input runs.
   *
   * @param scores  List of ScoredDocs instances to merge.
   * @param newLength Maximum number of documents to consider from each run for each topic (null for no limit).
   * @param topN     Maximum number of top documents to include in the merged run for each topic (null for no limit).
   * @return A new ScoredDocs instance containing the merged results.
   * @throws IllegalArgumentException if less than 2 runs are provided.
   */
  public static ScoredDocs merge(Map<QueryAndDoc, Double> scores, int newLength, int topN) {
    Instant start = Instant.now();

    List<Map.Entry<QueryAndDoc, Double>> scoreRank = new ArrayList<>(scores.entrySet());
    scoreRank.sort(
        Map.Entry.<QueryAndDoc, Double>comparingByKey(
          Comparator.comparing(QueryAndDoc::query)
        )
          .thenComparing(
          // Sort by descending score
            Map.Entry.<QueryAndDoc, Double>comparingByValue()
                .reversed())
            // Tie-break by doc ID, then shard index (like TopDocs#merge)
            .thenComparing(
                Map.Entry.<QueryAndDoc, Double>comparingByKey(
                    Comparator.comparing(QueryAndDoc::doc))));

    Instant end = Instant.now();
    Duration timeElapsed = Duration.between(start, end);
    System.out.println("Sorting map: "+ timeElapsed.toSeconds() +" seconds");
    start = end;
    ScoredDocs scoreDocs = new ScoredDocs();
    scoreDocs.lucene_documents = new Document[newLength];
    scoreDocs.lucene_docids = new int[newLength];
    scoreDocs.docids = new String[newLength];
    scoreDocs.scores = new float[newLength];
    int rank = 0;
    int index = 0;
    String curQuery = "";
    for (int i = 0; i < scoreRank.size(); i++) {
      Map.Entry<QueryAndDoc, Double> entry = scoreRank.get(i);
      String query = entry.getKey().query;
      if(query.equals(curQuery) && rank >= topN){
        continue;
      }
      else if(query.equals(curQuery)){
        rank++;
      }
      else{
        rank = 1;
        curQuery = query;
      }
      scoreDocs.docids[index] = entry.getKey().doc;
      scoreDocs.scores[index] = entry.getValue().floatValue();
      Document doc = new Document();
      doc.add(new StoredField(ScoredDocsFuser.TOPIC, query));
      scoreDocs.lucene_documents[index] = doc;
      scoreDocs.lucene_docids[index] = rank;
      index++;
    }
    end = Instant.now();
    timeElapsed = Duration.between(start, end);
    System.out.println("Reranking for topN: "+ timeElapsed.toSeconds() +" seconds");
    return scoreDocs;
  }

  public static Hit[] rrf(List<Hit[]> runs, int rrf_k, int depth, int k) {
    // Instant start = Instant.now();
    for (Hit[] run : runs) {
      HitFuser.rescoreRRF(rrf_k, run);
    }
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Rescoring runs: "+ timeElapsed.toSeconds() +" seconds");
    return HitFuser.merge(runs, depth, k);
  }

  // public record Hit(Document query, String docid, double score, int rank){}

  public static Hit[] rrf3(List<Hit[]> runs, int rrf_k, int depth, int topN){
    // Instant start = Instant.now();
    int newLength = 0; // find length for appended array
    for (Hit[] scoredDocs : runs){
      newLength += scoredDocs.length;
    }
    Hit[] scores = new Hit[newLength];
    int index = 0;
    for (Hit[] scoredDocs : runs){
      for(int i = 0; i < scoredDocs.length; i++){
        scores[index + i] = new Hit(scoredDocs[i]);
        scores[index + i].score = 1 / ((double)scoredDocs[i].rank + rrf_k);
        // System.out.println(scores[index + i].score);
        // scores[index + 1] = new Hit(scoredDocs.lucene_documents[i], scoredDocs.docids[i], 1 / (double)(scoredDocs.lucene_docids[i] + (float)rrf_k), 0);
      }
      index += scoredDocs.length;
    }
    // System.out.println("appended");
    Arrays.sort(scores, (s1, s2) -> {
      String topic1 = s1.query;
      String topic2 = s2.query;
      int topicComparison = (topic1.compareTo(topic2));
      if (topicComparison != 0) {
        return topicComparison;
      }
      int scoreComparison = Double.compare(s2.score, s1.score);
      int docComparison = s1.docid.compareTo(s2.docid);
      return docComparison != 0 ? docComparison : scoreComparison;
    });
    // System.out.println("sorted");
    int write = 0;
    for(int i = 1; i < scores.length; i++){
      if(scores[i].docid.equals(scores[i-1].docid) && scores[i].query.equals(scores[i-1].query)){
        scores[write].score += scores[i].score;
      }
      else{
        write++;
        scores[write] = new Hit(scores[i]);
      }
    }
    write++;
    // System.out.println(write);
    scores = Arrays.copyOf(scores, write);
    Arrays.sort(scores, (s1, s2) -> {
      String topic1 = s1.query;
      String topic2 = s2.query;
      int topicComparison = (topic1.compareTo(topic2));
      if (topicComparison != 0) {
        return topicComparison;
      }
      int scoreComparison = Double.compare(s2.score, s1.score);
      int docComparison = s1.docid.compareTo(s2.docid);
      return scoreComparison != 0 ? scoreComparison : docComparison;
    });
    // System.out.println("sorted");
    write = 0;
    int rank = 0;
    String curQuery = "";
    for(int i = 0; i < scores.length; i++){
      if(curQuery.equals(scores[i].query) && rank > topN){
        continue;
      }
      else if(curQuery.equals(scores[i].query)){
        rank++;
      }
      else{
        rank = 1;
        curQuery = scores[i].query;
      }
      scores[write] = new Hit(scores[i]);
      scores[write].rank = rank;
      write++;
    }
    // System.out.println(write);
    scores = Arrays.copyOf(scores, write);
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Total: "+ timeElapsed.toSeconds() +" seconds");
    return scores;
  }

  public static ScoredDocs rrf2(List<ScoredDocs> runs, int rrf_k, int depth, int topN){
    Instant start = Instant.now();
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
    Instant end = Instant.now();
    Duration timeElapsed = Duration.between(start, end);
    System.out.println("Appending runs: "+ timeElapsed.toSeconds() +" seconds");
    start = end;
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
    end = Instant.now();
    timeElapsed = Duration.between(start, end);
    System.out.println("Summing scores: "+ timeElapsed.toSeconds() +" seconds");
    start = end;
  
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
    end = Instant.now();
    timeElapsed = Duration.between(start, end);
    System.out.println("Sorting by score and limit top n: "+ timeElapsed.toSeconds() +" seconds");
    return summed;
  }

  /**
   * Process the fusion of ScoredDocs objects based on the specified method.
   *
   * @param runs List of ScoredDocs objects to be fused.
   * @throws IOException If an I/O error occurs while saving the output.
   */
  public void fuse(List<Hit[]> runs) throws IOException {
    // Instant start = Instant.now();
    Hit[] fusedRun;

    // Select fusion method
    switch (args.method.toLowerCase()) {
      // case LUCENE:
      //   fusedRun = reciprocalRankFusion(runs, args.rrf_k, args.depth, args.k);
      //   break;
      case METHOD_RRF:
        fusedRun = rrf(runs, args.rrf_k, args.depth, args.k);
        break;
      case APPEND:
        fusedRun = rrf3(runs, args.rrf_k, args.depth, args.k);
        break;
      // case METHOD_INTERPOLATION:
      //   fusedRun = interpolation(runs, args.alpha, args.depth, args.k);
      //   break;
      // case METHOD_AVERAGE:
      //   fusedRun = average(runs, args.depth, args.k);
      //   break;
      // case METHOD_NORMALIZE:
      //   fusedRun = normalize(runs, args.depth, args.k);
      //   break;
      default:
        throw new IllegalArgumentException("Unknown fusion method: " + args.method + 
            ". Supported methods are: average, rrf, interpolation.");
    }

    Path outputPath = Paths.get(args.output);
    HitFuser.saveToTxt(outputPath, args.runtag,  fusedRun);
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Total: "+ timeElapsed.toSeconds() +" seconds");
  }
}
