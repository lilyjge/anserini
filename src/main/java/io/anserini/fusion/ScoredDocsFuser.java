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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
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
import java.time.Duration;
import java.time.Instant;
public class ScoredDocsFuser {
  public static final String TOPIC = "TOPIC";

  /**
   * Reads a TREC run file and returns a ScoredDocs containing the data.
   * 
   * @param filepath Path to the TREC run file.
   * @throws IOException If the file cannot be read.
   * @return A ScoredDocs object containing the data from the TREC run file.
   */
  public static ScoredDocs readRun(Path filepath, boolean reSort) throws IOException {
    ScoredDocs scoredDocs = new ScoredDocs();
    try (BufferedReader br = new BufferedReader(new FileReader(filepath.toFile()))) {
      List<Document> lucene_documents = new ArrayList<>(); // topic
      List<String> docids = new ArrayList<>(); // docid
      List<Float> scores = new ArrayList<>(); // score
      List<Integer> rank = new ArrayList<>(); // rank

      String line;
      while ((line = br.readLine()) != null) {
        String[] data = line.split("\\s+");
  
        // Populate the lists with the parsed topic and docid
        Document doc = new Document();
        doc.add(new StoredField(TOPIC, data[0]));
        lucene_documents.add(doc);
        docids.add(data[2]);

        // Parse RANK as integer
        int rankInt = Integer.parseInt(data[3]);
        rank.add(rankInt);

        // Parse SCORE as float
        float scoreFloat = Float.parseFloat(data[4]);
        scores.add(scoreFloat);
      }

      scoredDocs.lucene_documents = lucene_documents.toArray(new Document[0]);
      scoredDocs.docids = docids.toArray(new String[0]);
      scoredDocs.scores = ArrayUtils.toPrimitive(scores.toArray(new Float[scores.size()]), Float.NaN);
      scoredDocs.lucene_docids = ArrayUtils.toPrimitive(rank.toArray(new Integer[0]));
    }
  
    if (reSort) {
      ScoredDocsFuser.sortScoredDocs(scoredDocs, true, 0);
    }

    return scoredDocs;
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
  public static void rescore(RescoreMethod method, int rrfK, double scale, ScoredDocs scoredDocs) {
    switch (method) {
      case RRF -> ScoredDocsFuser.rescoreRRF(rrfK, scoredDocs);
      case SCALE -> ScoredDocsFuser.rescoreScale(scale, scoredDocs);
      case NORMALIZE -> ScoredDocsFuser.normalizeScores(scoredDocs);
      default -> throw new UnsupportedOperationException("Unknown rescore method: " + method);
    }
  }

  private static void rescoreRRF(int rrfK, ScoredDocs scoredDocs) {
    int length = scoredDocs.lucene_documents.length;
    for (int i = 0; i < length; i++) {
      float score = (float)(1.0 / (rrfK + scoredDocs.lucene_docids[i]));
      scoredDocs.scores[i] = score;
    }
  }

  private static void rescoreScale(double scale, ScoredDocs scoredDocs) {
    int length = scoredDocs.lucene_documents.length;
    for (int i = 0; i < length; i++) {
      float score = (float) (scoredDocs.scores[i] * scale);
      scoredDocs.scores[i] = score;
    }
  }

  private static void normalizeScores(ScoredDocs scoredDocs) {
    Map<String, List<Integer>> indicesForTopics = new HashMap<String, List<Integer>>(); // topic, list of indices for that topic
    int length = scoredDocs.lucene_documents.length;
    for (int i = 0; i < length; i++) {
      indicesForTopics.computeIfAbsent(scoredDocs.lucene_documents[i].get(TOPIC), k -> new ArrayList<>()).add(i);
    }

    for (List<Integer> topicIndices : indicesForTopics.values()) {
      int numRecords = topicIndices.size();
      float minScore = scoredDocs.scores[topicIndices.get(0)];
      float maxScore = scoredDocs.scores[topicIndices.get(numRecords - 1)];
      for (int i = 0; i < numRecords; i++) {
        int index = topicIndices.get(i);
        minScore = Float.min(minScore, scoredDocs.scores[index]);
        maxScore = Float.max(maxScore, scoredDocs.scores[index]);
      }

      for (int i = 0; i < numRecords; i++) {
        int index = topicIndices.get(i);
        float normalizedScore = ((float) scoredDocs.scores[index] - minScore) / (maxScore - minScore);
        scoredDocs.scores[index] = normalizedScore;
      }
    }
  }

  /**
   * Sorts given ScoredDocs by topic, then by score.
   * 
   * @param scoredDocs ScoredDocs object to be sorted.
   */
  public static int sortScoredDocs(ScoredDocs scoredDocs, Boolean scoreFirst, int topN){
    Integer[] indices = new Integer[scoredDocs.lucene_documents.length];
    for (int i = 0; i < indices.length; i++) {
      indices[i] = i;
    }

    Arrays.sort(indices, (index1, index2) -> {
      String topic1 = scoredDocs.lucene_documents[index1].get(TOPIC);
      String topic2 = scoredDocs.lucene_documents[index2].get(TOPIC);
      int topicComparison = (topic1.compareTo(topic2));
      if (topicComparison != 0) {
        return topicComparison;
      }
      int scoreComparison = Float.compare(scoredDocs.scores[index2], scoredDocs.scores[index1]);
      int docComparison = scoredDocs.docids[index1].compareTo(scoredDocs.docids[index2]);
      if(scoreFirst){
        return scoreComparison != 0 ? scoreComparison : docComparison;
      }
      else{
        return docComparison != 0 ? docComparison : scoreComparison;
      }
    });

    Document[] sorted_lucene_documents = new Document[indices.length];
    String[] sortedDocids = new String[indices.length];
    float[] sortedScores = new float[indices.length];
    int[] sortedRanks = new int[indices.length];
    String curQuery = "";
    int rank = 0;
    int overflow = 0;
    for (int i = 0; i < indices.length; i++) {
      int index = indices[i];
      sorted_lucene_documents[i] = scoredDocs.lucene_documents[index];
      sortedDocids[i] = scoredDocs.docids[index];
      sortedScores[i] = scoredDocs.scores[index];
      if(!curQuery.equals(scoredDocs.lucene_documents[index].get(TOPIC))){
        rank = 1;
        curQuery = scoredDocs.lucene_documents[index].get(TOPIC);
      }
      else{
        rank++;
      }
      sortedRanks[i] = rank;
      if(rank > topN){
        overflow++;
      }
    }

    scoredDocs.lucene_documents = sorted_lucene_documents;
    scoredDocs.docids = sortedDocids;
    scoredDocs.scores = sortedScores;
    scoredDocs.lucene_docids = sortedRanks;
    // System.out.println(overflow);
    return overflow;
  }

  public static ScoredDocs merge(List<ScoredDocs> runs, Integer depth, Integer k) {
    Instant start = Instant.now();
    if (runs.size() < 2) {
      throw new IllegalArgumentException("Merge requires at least 2 runs.");
    }

    // for every topic, produce a map of docid to score, num of accumulated
    HashMap<String, HashMap<String, AbstractMap.SimpleEntry<Float, Integer>>> docScores = new HashMap<>();
    for (ScoredDocs run : runs) {
      for (int i = 0; i < run.lucene_documents.length; i++) {
        String query = run.lucene_documents[i].get(TOPIC);
        String docid = run.docids[i];
        Float score = run.scores[i];
        docScores.computeIfAbsent(query, key -> new HashMap<>())
          .merge(docid, new AbstractMap.SimpleEntry<>(score, 1), 
                (existing, newValue) -> 
                  existing.getValue() >= depth ? existing : new AbstractMap.SimpleEntry<>(existing.getKey() + newValue.getKey(), existing.getValue() + 1));
      }
    }
    Instant end = Instant.now();
    Duration timeElapsed = Duration.between(start, end);
    System.out.println("Accumulating scores: "+ timeElapsed.toSeconds() +" seconds");
    start = end;
    
    List<Document> lucene_documents = new ArrayList<>(); // topic
    List<String> docids = new ArrayList<>(); // docid
    List<Float> score = new ArrayList<>(); // score
    List<Integer> rank = new ArrayList<>(); // rank
    for (String query : docScores.keySet()) {
      // for the current query, a list of all docids and scores, sorted by scores
      List<Map.Entry<String, Float>> sortedDocScores = docScores.get(query).entrySet().stream()
        .map(entry -> Map.entry(entry.getKey(), entry.getValue().getKey()))
        .sorted(Map.Entry.<String, Float>comparingByValue().reversed().thenComparing(Map.Entry.<String, Float>comparingByKey()))
        .limit(k != null ? k : Integer.MAX_VALUE)
        .collect(Collectors.toList());

      for (int i = 0; i < sortedDocScores.size(); i++) {
        Map.Entry<String, Float> entry = sortedDocScores.get(i);
        Document doc = new Document();
        doc.add(new StoredField(TOPIC, query));
        lucene_documents.add(doc);
        docids.add(entry.getKey());
        rank.add(i + 1);
        score.add(entry.getValue());
      }
    }

    ScoredDocs mergedRun = new ScoredDocs();
    mergedRun.lucene_documents = lucene_documents.toArray(new Document[0]);
    mergedRun.docids = docids.toArray(new String[0]);
    mergedRun.scores = ArrayUtils.toPrimitive(score.toArray(new Float[score.size()]), Float.NaN);
    mergedRun.lucene_docids = ArrayUtils.toPrimitive(rank.toArray(new Integer[0]));
    end = Instant.now();
    timeElapsed = Duration.between(start, end);
    System.out.println("Sort and limit: "+ timeElapsed.toSeconds() +" seconds");
    return mergedRun;
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
  public static void saveToTxt(Path outputPath, String tag, ScoredDocs run) throws IOException {
    if (run.lucene_documents == null || run.lucene_documents.length == 0) {
      throw new IllegalStateException("Nothing to save. ScoredDocs is empty");
    }

    // ScoredDocsFuser.sortScoredDocs(run);
    try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
      for (int i = 0; i < run.lucene_documents.length; i++) {
        writer.write(String.format("%s Q0 %s %d %.6f %s%n", 
          run.lucene_documents[i].get(TOPIC), run.docids[i], run.lucene_docids[i], run.scores[i], tag));
      }
    }
  }
}
