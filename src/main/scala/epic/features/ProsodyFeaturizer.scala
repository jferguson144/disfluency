package epic.features

import epic.framework.Feature
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature
import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import scala.collection.mutable.ArrayBuffer

class ProsodyFeaturizer private (prosodyLookupTable: ProsodyFeaturizer.ProsodyLookupTable) extends SplitSpanFeaturizer[String] with Serializable {
  final val UNKNOWN_FEATURE_VAL: Double = -1.0
  final val NUM_FEATURES = 5
  final val UNKNOWN_FEATURES: Array[Double] = Array.fill(NUM_FEATURES) { UNKNOWN_FEATURE_VAL }

  def anchor(w: IndexedSeq[String]): SplitSpanFeatureAnchoring[String] = new SplitSpanFeatureAnchoring[String] {
    val prosodyFeats: IndexedSeq[Array[Double]] = if (!prosodyLookupTable.contains(w)) {
      println("Sentence wasn't found in lookup table: " + w.mkString(" "))
      (0 until w.size).map(i => UNKNOWN_FEATURES); //Some non-value for each position we're considering breaks for (split, prior to beginning, after end)
    } else {
      prosodyLookupTable(w)
    }
    def featuresForSplit(begin: Int, split: Int, end: Int): Array[Feature] = {
      val featsBeforeBegin = if (begin == 0) { UNKNOWN_FEATURES } else { prosodyFeats(begin - 1) }
      val featsAfterBegin = prosodyFeats(begin)
      val featsBeforeEnd = prosodyFeats(end - 1)
      val featsAfterEnd = if (end == prosodyFeats.size) {UNKNOWN_FEATURES} else {prosodyFeats(end)}

      def subtractWithUnknown(v1: Double, v2: Double): String = {
        if (v1 == UNKNOWN_FEATURE_VAL || v2 == UNKNOWN_FEATURE_VAL) {
          return "x"
        } else {
          if (v1 > v2) {
            return "+"
          } else {
            return "-"
          }
        }
      }

      val rateDiffBegin = subtractWithUnknown(featsAfterBegin(0), featsBeforeBegin(0))
      val rateDiffEnd = subtractWithUnknown(featsAfterEnd(0), featsBeforeEnd(0))
      val pitchDiffBegin = subtractWithUnknown(featsAfterBegin(1), featsBeforeBegin(1))
      val pitchDiffEnd = subtractWithUnknown(featsAfterEnd(1), featsBeforeEnd(1))
      val energyDiffBegin = subtractWithUnknown(featsAfterBegin(2), featsBeforeBegin(2))
      val energyDiffEnd = subtractWithUnknown(featsAfterEnd(2), featsBeforeEnd(2))

      //Not so sure that duration difference will be at all meaningful (essentially just word length diff)
      val durationDiffBegin = subtractWithUnknown(featsAfterBegin(3), featsBeforeBegin(3))
      val durationDiffEnd = subtractWithUnknown(featsAfterEnd(3), featsBeforeEnd(3))

      def binPause(pauseDuration: Double): String = {
        if (pauseDuration == 0) {
          "0"
        } else {
          "1"
        } 
      }
      val pauseBegin = if (featsBeforeBegin(4) == UNKNOWN_FEATURE_VAL) { "x" } else { binPause(featsBeforeBegin(4)) }
      val pauseEnd = if (end == prosodyFeats.length - 1) {
        "e"
      } else if (featsBeforeEnd(4) == UNKNOWN_FEATURE_VAL) {
        "x"
      } else {
        binPause(featsBeforeEnd(4))
      }

      val features = new ArrayBuffer[Feature]
      features.append(new IndicatorFeature(("BEGIN_PAUSE", pauseBegin)))
      features.append(new IndicatorFeature(("END_PAUSE", pauseEnd)))
      var numPauses = 0
      for (i <- begin until end-1) {
        if (prosodyFeats(i)(4) > 0) {
          numPauses += 1
        }
      }
      features.append(new IndicatorFeature(("NUM_PAUSES", numPauses)))
      features.append(new IndicatorFeature(("NUM_PAUSES_LENGTH", numPauses, end-begin)))
      features.toArray
    }

    def featuresForSpan(begin: Int, end: Int): Array[Feature] = Array.empty[Feature]

    def words: IndexedSeq[String] = w
  }
}
object ProsodyFeaturizer {

  type ProsodyLookupTable = HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]];

  def makeLookupTable(pathToTaggedSentences: String): ProsodyLookupTable = {
    val in = breeze.io.FileStreams.input(new File(pathToTaggedSentences))
    val br = new BufferedReader(new InputStreamReader(in, "UTF-8"));
    val lookupTable = new HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]]
    var thisSent = new ArrayBuffer[String];
    var thisSentFeats = new ArrayBuffer[Array[Double]];
    while (br.ready()) {
      val line = br.readLine();
      if (line.trim.isEmpty) {
        lookupTable.put(thisSent, thisSentFeats);
        thisSent = new ArrayBuffer[String];
        thisSentFeats = new ArrayBuffer[Array[Double]];
      } else {
        val splitLine = line.trim.split("\\s+");
        if (splitLine.size != 2) {
          println("WARNING: Bad line, split into more than two parts on whitespace: " + splitLine);
        }
        thisSent += splitLine(0)
        val feats = splitLine(1).split(",");
        val featsArray = new ArrayBuffer[Double];
        for (f <- feats) {
          featsArray += f.toDouble
        }
        thisSentFeats += featsArray.toArray
      }
    }
    if (!thisSent.isEmpty) {
      lookupTable.put(thisSent, thisSentFeats);
    }
    println("Loaded " + lookupTable.size + " entries from " + pathToTaggedSentences);
    lookupTable;
  }

  def apply(pathsToTaggedSentences: Seq[String]) = {
    val lookupTable = pathsToTaggedSentences.map(makeLookupTable(_)).reduce(_ ++ _);
    new ProsodyFeaturizer(lookupTable);
  }

}
