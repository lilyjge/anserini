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

package io.anserini.index.generator;

import io.anserini.collection.SourceDocument;
import io.anserini.index.Constants;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BytesRef;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * A document generator for creating Lucene documents with Parquet dense
 * vector data.
 * Implements the LuceneDocumentGenerator interface.
 *
 * @param <T> the type of SourceDocument
 */
public class DenseVectorDocumentGenerator<T extends SourceDocument> implements LuceneDocumentGenerator<T> {
  private static final Logger LOG = LogManager.getLogger(DenseVectorDocumentGenerator.class);

  /**
   * Creates a Lucene document from the source document.
   *
   * @param src the source document
   * @return the created Lucene document
   * @throws InvalidDocumentException if the document is invalid
   */
  @Override
  public Document createDocument(T src) throws InvalidDocumentException {

    try {
      float[] contents = src.vector();
      if (contents == null || contents.length == 0) {
        LOG.error("Vector data is null or empty for document ID: " + src.id());
        throw new InvalidDocumentException();
      }

      // Create and populate the Lucene document
      final Document document = new Document();
      document.add(new StringField(Constants.ID, src.id(), Field.Store.YES));
      document.add(new BinaryDocValuesField(Constants.ID, new BytesRef(src.id())));
      document.add(new KnnFloatVectorField(Constants.VECTOR, contents, VectorSimilarityFunction.DOT_PRODUCT));

      return document;

    } catch (InvalidDocumentException e) {
      throw e;
    } catch (Exception e) {
      LOG.error("Unexpected error creating document for ID: " + src.id(), e);
      throw new InvalidDocumentException();
    }
  }
}
