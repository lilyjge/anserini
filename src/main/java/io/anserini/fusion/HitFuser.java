package io.anserini.fusion;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


public class HitFuser {
    /**
   * Reads a TREC run file and returns a ScoredDocs containing the data.
   * 
   * @param filepath Path to the TREC run file.
   * @throws IOException If the file cannot be read.
   * @return A ScoredDocs object containing the data from the TREC run file.
   */
  public static Hit[] readRun(Path filepath, boolean reSort) throws IOException {
    Hit[] run;
    try (BufferedReader br = new BufferedReader(new FileReader(filepath.toFile()))) {
      List<Hit> hits = new ArrayList<>();

      String line;
      while ((line = br.readLine()) != null) {
        String[] data = line.split("\\s+");

        hits.add(new Hit(data[0], data[2], Integer.parseInt(data[3]), Double.parseDouble(data[4])));
      }
      run = hits.toArray(new Hit[0]);
    }
  
    // if (reSort) {
    //   ScoredDocsFuser.sortScoredDocs(scoredDocs, true, 0);
    // }

    return run;
  }

  public static void norm(Hit[] scoredDocs){
    Map<String, List<Integer>> indicesForTopics = new HashMap<String, List<Integer>>(); // topic, list of indices for that topic
    for (int i = 0; i < scoredDocs.length; i++) {
      indicesForTopics.computeIfAbsent(scoredDocs[i].query, k -> new ArrayList<>()).add(i);
    }

    for (List<Integer> topicIndices : indicesForTopics.values()) {
      int numRecords = topicIndices.size();
      double minScore = scoredDocs[topicIndices.get(0)].score;
      double maxScore = scoredDocs[topicIndices.get(numRecords - 1)].score;
      for (int i = 0; i < numRecords; i++) {
        int index = topicIndices.get(i);
        minScore = Double.min(minScore, scoredDocs[index].score);
        maxScore = Double.max(maxScore, scoredDocs[index].score);
      }

      for (int i = 0; i < numRecords; i++) {
        int index = topicIndices.get(i);
        scoredDocs[index].score = (scoredDocs[index].score - minScore) / (maxScore - minScore);
      }
    }
  }

  public static void rescoreRRF(int rrfK, Hit[] scoredDocs) {
    for (int i = 0; i < scoredDocs.length; i++) {
      scoredDocs[i].score = 1.0 / (rrfK + (double)scoredDocs[i].rank);
    }
  }

  public static void scale(double scale, Hit[] scoredDocs) {
    for (int i = 0; i < scoredDocs.length; i++) {
      scoredDocs[i].score = scale * scoredDocs[i].score;
    }
  }

  public static Hit[] merge(List<Hit[]> runs, Integer depth, Integer k) {
    // Instant start = Instant.now();
    if (runs.size() < 2) {
      throw new IllegalArgumentException("Merge requires at least 2 runs.");
    }

    // for every topic, produce a map of docid to score, num of accumulated
    HashMap<String, HashMap<String, AbstractMap.SimpleEntry<Double, Integer>>> docScores = new HashMap<>();
    for (Hit[] run : runs) {
      for (int i = 0; i < run.length; i++) {
        String query = run[i].query;
        String docid = run[i].docid;
        double score = run[i].score;
        docScores.computeIfAbsent(query, key -> new HashMap<>())
          .merge(docid, new AbstractMap.SimpleEntry<>(score, 1), 
                (existing, newValue) -> 
                  existing.getValue() >= depth ? existing : new AbstractMap.SimpleEntry<>(existing.getKey() + newValue.getKey(), existing.getValue() + 1));
      }
    }
    // Instant end = Instant.now();
    // Duration timeElapsed = Duration.between(start, end);
    // System.out.println("Accumulating scores: "+ timeElapsed.toSeconds() +" seconds");
    // start = end;
    
    List<Hit> merged = new ArrayList<>();
    for (String query : docScores.keySet()) {
      // for the current query, a list of all docids and scores, sorted by scores
      List<Map.Entry<String, Double>> sortedDocScores = docScores.get(query).entrySet().stream()
        .map(entry -> Map.entry(entry.getKey(), entry.getValue().getKey()))
        .sorted(Map.Entry.<String, Double>comparingByValue().reversed().thenComparing(Map.Entry.<String, Double>comparingByKey()))
        .limit(k != null ? k : Integer.MAX_VALUE)
        .collect(Collectors.toList());

      for (int i = 0; i < sortedDocScores.size(); i++) {
        Map.Entry<String, Double> entry = sortedDocScores.get(i);
        merged.add(new Hit(query, entry.getKey(), i + 1, entry.getValue()));
      }
    }

    Hit[] mergedRun = merged.toArray(new Hit[0]);
    // end = Instant.now();
    // timeElapsed = Duration.between(start, end);
    // System.out.println("Sort and limit: "+ timeElapsed.toSeconds() +" seconds");
    return mergedRun;
  }

  public static void saveToTxt(Path outputPath, String tag, Hit[] run) throws IOException {
    if (run.length == 0) {
      throw new IllegalStateException("Nothing to save. ScoredDocs is empty");
    }

    // ScoredDocsFuser.sortScoredDocs(run);
    try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
      for (int i = 0; i < run.length; i++) {
        writer.write(String.format("%s Q0 %s %d %.6f %s%n", 
          run[i].query, run[i].docid, run[i].rank, run[i].score, tag));
      }
    }
  }
}
