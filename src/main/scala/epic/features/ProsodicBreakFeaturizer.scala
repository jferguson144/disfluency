package epic.features

import epic.framework.Feature
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature
import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import scala.collection.mutable.ArrayBuffer
import epic.features.PreprocessedFeaturizer
import breeze.collection.mutable.TriangularArray

class ProsodicBreakFeaturizer (breakLookupTable: ProsodicBreakFeaturizer.BreakLookupTable) extends SurfaceFeaturizer[String] with Serializable {
  final val BEGIN_SENTENCE_BREAK = "4"
  final val UNKNOWN_BREAK = "x"
  def anchor(w: IndexedSeq[String]): SurfaceFeatureAnchoring[String] = new SurfaceFeatureAnchoring[String] {
    val prosodyFeats = if (!breakLookupTable.contains(w.mkString(" "))) {
      println("Sentence wasn't found in lookup table: " + w.mkString(" "))
      (0 until w.size).map(i => UNKNOWN_BREAK); //Some non-value for each position we're considering breaks for (split, prior to beginning, after end)
    } else {
      breakLookupTable(w.mkString(" "))
    }
    val feats = (0 until w.size).map(
      i =>{ prosodyFeats(i) })

    def featuresForSpan(begin: Int, end: Int): Array[Feature] = {
      val ret = ArrayBuffer[Feature]()
      if (end-begin > 1) {
        ret.append(IndicatorFeature(("B", feats(begin))))
        for (i <- begin+1 until end-1) {
          ret.append(IndicatorFeature(("I",feats(i))))
        }
        ret.append(IndicatorFeature(("E", feats(end-1))))
      } else {
        ret.append(IndicatorFeature(("S", feats(begin))))
      }
      ret.toArray
    }

    def words: IndexedSeq[String] = w
  }
}

object ProsodicBreakFeaturizer {

  type BreakLookupTable = HashMap[String, IndexedSeq[String]];

  def makeLookupTable(pathToTaggedSentences: String): BreakLookupTable = {
    val in = breeze.io.FileStreams.input(new File(pathToTaggedSentences))
    val br = new BufferedReader(new InputStreamReader(in, "UTF-8"));
    val lookupTable = new HashMap[String, IndexedSeq[String]]
    while (br.ready()) {
      val sentence = br.readLine().trim()
      val breaks = br.readLine().trim()
      br.readLine()
      lookupTable.put(sentence, breaks.split(" "))
    }
    println("Loaded " + lookupTable.size + " entries from " + pathToTaggedSentences);
    lookupTable;
  }

  def apply(pathToTaggedSentences: String) = {
    val lookupTable = makeLookupTable(pathToTaggedSentences);
    new ProsodicBreakFeaturizer(lookupTable);
  }

}
