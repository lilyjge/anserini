package io.anserini.fusion;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;

import io.anserini.search.ScoredDocs;

//replace topic wtih const
public class HitsFuser {

  /**
   * Reads a TREC run file and returns a ScoredDocs containing the data.
   * 
   * @param filepath Path to the TREC run file.
   * @throws IOException If the file cannot be read.
   * @return A ScoredDocs object containing the data from the TREC run file.
   */
  public static Hits readRun(Path filepath, boolean reSort) throws IOException {
    Hits hits;
    try (BufferedReader br = new BufferedReader(new FileReader(filepath.toFile()))) {
      ArrayList<String> queries = new ArrayList<>(); // topic
      ArrayList<String> docids = new ArrayList<>(); // docid
      ArrayList<Double> scores = new ArrayList<>(); // score
      ArrayList<Integer> rank = new ArrayList<>(); // rank

      String line;
      while ((line = br.readLine()) != null) {
        String[] data = line.split("\\s+");
  
        // Populate the lists with the parsed topic and docid
        queries.add(data[0]);
        docids.add(data[2]);

        // Parse RANK as integer
        int rankInt = Integer.parseInt(data[3]);
        rank.add(rankInt);

        // Parse SCORE as float
        double scoreFloat = Double.parseDouble(data[4]);
        scores.add(scoreFloat);
      }

      hits = new Hits(queries, docids, scores, rank);
    }
  
    if (reSort) {
      HitsFuser.sortHits(hits, false);
    }

    return hits;
  }

  /**
   * Rescored given ScoredDocs using the specified method.
   *
   * @param method  Rescore method to be applied (e.g., RRF, SCALE, NORMALIZE).
   * @param rrfK    Parameter k needed for reciprocal rank fusion.
   * @param scale   Scaling factor needed for rescoring by scaling.
   * @param scoredDocs ScoredDocs object to be rescored.
   * @throws UnsupportedOperationException If an unsupported rescore method is provided.
   */
  public static void rescore(RescoreMethod method, int rrfK, double scale, Hits scoredDocs) {
    switch (method) {
      case RRF -> rescoreRRF(rrfK, scoredDocs);
      case SCALE -> rescoreScale(scale, scoredDocs);
      case NORMALIZE -> normalizeScores(scoredDocs);
      default -> throw new UnsupportedOperationException("Unknown rescore method: " + method);
    }
  }

  private static void rescoreRRF(int rrfK, Hits scoredDocs) {
    int length = scoredDocs.query.length;
    for (int i = 0; i < length; i++) {
      float score = (float)(1.0 / (rrfK + scoredDocs.rank[i]));
      scoredDocs.score[i] = score;
    }
  }

  private static void rescoreScale(double scale, Hits scoredDocs) {
    int length = scoredDocs.query.length;
    for (int i = 0; i < length; i++) {
      float score = (float) (scoredDocs.score[i] * scale);
      scoredDocs.score[i] = score;
    }
  }

  private static void normalizeScores(Hits scoredDocs) {
    Map<String, List<Integer>> indicesForTopics = new HashMap<String, List<Integer>>(); // topic, list of indices for that topic
    int length = scoredDocs.query.length;
    for (int i = 0; i < length; i++) {
      indicesForTopics.computeIfAbsent(scoredDocs.query[i], k -> new ArrayList<>()).add(i);
    }

    for (List<Integer> topicIndices : indicesForTopics.values()) {
      int numRecords = topicIndices.size();
      double minScore = scoredDocs.score[topicIndices.get(0)];
      double maxScore = scoredDocs.score[topicIndices.get(numRecords - 1)];
      for (int i = 0; i < numRecords; i++) {
        int index = topicIndices.get(i);
        minScore = Double.min(minScore, scoredDocs.score[index]);
        maxScore = Double.max(maxScore, scoredDocs.score[index]);
      }

      for (int i = 0; i < numRecords; i++) {
        int index = topicIndices.get(i);
        double normalizedScore = (scoredDocs.score[index] - minScore) / (maxScore - minScore);
        scoredDocs.score[index] = normalizedScore;
      }
    }
  }

  /**
   * Merges multiple ScoredDocs instances into a single ScoredDocs instance.
   * The merged ScoredDocs will contain the top documents for each topic, with scores summed across the input runs.
   *
   * @param runs  List of ScoredDocs instances to merge.
   * @param depth Maximum number of documents to consider from each run for each topic (null for no limit).
   * @param k     Maximum number of top documents to include in the merged run for each topic (null for no limit).
   * @return A new ScoredDocs instance containing the merged results.
   * @throws IllegalArgumentException if less than 2 runs are provided.
   */
  public static Hits merge(List<Hits> runs, Integer depth, Integer k) {
    if (runs.size() < 2) {
      throw new IllegalArgumentException("Merge requires at least 2 runs.");
    }

    // for every topic, produce a map of docid to score, num of accumulated
    HashMap<String, HashMap<String, AbstractMap.SimpleEntry<Double, Integer>>> docScores = new HashMap<>();
    for (Hits run : runs) {
      for (int i = 0; i < run.query.length; i++) {
        String query = run.query[i];
        String docid = run.docid[i];
        double score = run.score[i];
        docScores.computeIfAbsent(query, key -> new HashMap<>())
          .merge(docid, new AbstractMap.SimpleEntry<>(score, 1), 
                (existing, newValue) -> 
                  existing.getValue() >= depth ? existing : new AbstractMap.SimpleEntry<>(existing.getKey() + newValue.getKey(), existing.getValue() + 1));
      }
    }
    
    ArrayList<String> queries = new ArrayList<>(); // topic
    ArrayList<String> docids = new ArrayList<>(); // docid
    ArrayList<Double> score = new ArrayList<>(); // score
    ArrayList<Integer> rank = new ArrayList<>(); // rank
    for (String query : docScores.keySet()) {
      // for the current query, a list of all docids and scores, sorted by scores
      List<Map.Entry<String, Double>> sortedDocScores = docScores.get(query).entrySet().stream()
        .map(entry -> Map.entry(entry.getKey(), entry.getValue().getKey()))
        .sorted(Map.Entry.<String, Double>comparingByValue().reversed().thenComparing(Map.Entry.<String, Double>comparingByKey()))
        .limit(k != null ? k : Integer.MAX_VALUE)
        .collect(Collectors.toList());

      for (int i = 0; i < sortedDocScores.size(); i++) {
        Map.Entry<String, Double> entry = sortedDocScores.get(i);
        queries.add(query);
        docids.add(entry.getKey());
        rank.add(i + 1);
        score.add(entry.getValue());
      }
    }
    Hits hits = new Hits(queries, docids, score, rank);

    return hits;
  }

  /**
   * Sorts given ScoredDocs by topic, then by score.
   * 
   * @param scoredDocs ScoredDocs object to be sorted.
   */
  public static void sortHits(Hits scoredDocs, boolean byDocs){
    Integer[] indices = new Integer[scoredDocs.query.length];
    for (int i = 0; i < indices.length; i++) {
      indices[i] = i;
    }

    Arrays.sort(indices, (index1, index2) -> {
      String topic1 = scoredDocs.query[index1];
      String topic2 = scoredDocs.query[index2];
      int topicComparison = (topic1.compareTo(topic2));
      if (topicComparison != 0) {
        return topicComparison;
      }
      int scoreComparison = Double.compare(scoredDocs.score[index2], scoredDocs.score[index1]);
      int docComparison = scoredDocs.docid[index1].compareTo(scoredDocs.docid[index2]);
      if(byDocs) return docComparison != 0 ? docComparison : scoreComparison;
      return scoreComparison != 0 ? scoreComparison : docComparison;
    });

    String[] sortedQueries = new String[indices.length];
    String[] sortedDocids = new String[indices.length];
    double[] sortedScores = new double[indices.length];
    int[] sortedRanks = new int[indices.length];
    for (int i = 0; i < indices.length; i++) {
      int index = indices[i];
      sortedQueries[i] = scoredDocs.query[index];
      sortedDocids[i] = scoredDocs.docid[index];
      sortedScores[i] = scoredDocs.score[index];
      sortedRanks[i] = scoredDocs.rank[index];
    }

    scoredDocs.query = sortedQueries;
    scoredDocs.docid = sortedDocids;
    scoredDocs.score = sortedScores;
    scoredDocs.rank = sortedRanks;
  }

  /**
   * Saves a ScoredDocs run data to a text file in the TREC run format.
   * 
   * @param outputPath Path to the output file.
   * @param tag Tag to be added to each record in the TREC run file. If null, the existing tags are retained.
   * @param run ScoredDocs object to be saved.
   * @throws IOException If an I/O error occurs while writing to the file.
   * @throws IllegalStateException If the ScoredDocs is empty.
   */
  public static void saveToTxt(Path outputPath, String tag, Hits run) throws IOException {
    if (run.query == null || run.query.length == 0) {
      throw new IllegalStateException("Nothing to save. ScoredDocs is empty");
    }

    HitsFuser.sortHits(run, false);
    try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
      for (int i = 0; i < run.query.length; i++) {
        writer.write(String.format("%s Q0 %s %d %.6f %s%n", 
          run.query[i], run.docid[i], run.rank[i], run.score[i], tag));
      }
    }
  }
}