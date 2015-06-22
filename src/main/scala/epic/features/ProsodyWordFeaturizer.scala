package epic.features

import epic.framework.Feature
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature
import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import scala.collection.mutable.ArrayBuffer
import breeze.collection.mutable.TriangularArray


class ProsodyWordFeaturizer (prosodyLookupTable: ProsodySpanFeaturizer.ProsodyLookupTable) extends WordFeaturizer[String] with Serializable {

  final val UNKNOWN_FEATURE_VAL: Double = -1.0
  final val NUM_FEATURES = 3
  final val UNKNOWN_FEATURES: Array[Double] = Array.fill(NUM_FEATURES) { UNKNOWN_FEATURE_VAL }
  val DURATION_THRESHOLD = 0.3
  val DURATION_BUCKET = 0.1
  val PAUSE_THRESHOLD = 0.5
  val ENERGY_THRESHOLD = 0.5
  val durationIndex = 0
  val pauseIndex = 1
  val energyIndex = 2
  
  
  
  val averageDurations = HashMap[String, Double]()
  val wordCounts = HashMap[String, Integer]()
  
  for ((k, v) <- prosodyLookupTable) {
    assert(k.length == v.length)
    for (i <- 0 until k.length) {
      val word = k(i)
      val duration = v(i)(durationIndex)
      val curTotal = averageDurations.getOrElse(word, 0.0)
      val curCount = wordCounts.getOrElse(word, 0).asInstanceOf[Integer]
      averageDurations.put(word, curTotal+duration)
      wordCounts.put(word, curCount+1)
    }
  }
  for ((word, total) <- averageDurations) {
    val count = wordCounts(word)
    averageDurations.put(word, total/count)
  }
  
  def anchor(w: IndexedSeq[String]): WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
    val prosodyFeats: IndexedSeq[Array[Double]] = if (!prosodyLookupTable.contains(w)) {
      println("Sentence wasn't found in lookup table: " + w.mkString(" "))
      (0 until w.size).map(i => UNKNOWN_FEATURES); //Some non-value for each position we're considering breaks for (split, prior to beginning, after end)
    } else {
      prosodyLookupTable(w)
    }
    var totalEnergy = 0
    var knownCount = 0
    for (i <- 0 until w.length) {
      val energy = prosodyFeats(i)(energyIndex)
      if (energy != -1) {
        totalEnergy += 1
        knownCount += 1
      }
    }
    val avgEnergy = totalEnergy/knownCount.toDouble
    
    val allFeatures = ArrayBuffer[Array[Feature]]()
    for (i <- 0 until w.length) {
      val begin = i
      val end = i+1
      val featsBeforeBegin = if (begin == 0) { UNKNOWN_FEATURES } else { prosodyFeats(begin - 1) }
      val featsAfterBegin = prosodyFeats(begin)
      
      def binPause(pauseDuration: Double): String = {
        if (pauseDuration == 0) {
          "0"
        } else {
          "1"
        } 
      }
      val pauseBegin = if (featsBeforeBegin(pauseIndex) == UNKNOWN_FEATURE_VAL) { "x" } else { binPause(featsBeforeBegin(pauseIndex)) }
      val pauseEnd = if (featsAfterBegin(pauseIndex) == UNKNOWN_FEATURE_VAL) { "x" } else { binPause(featsAfterBegin(pauseIndex)) }

      val features = new ArrayBuffer[Feature]
      features.append(new IndicatorFeature(("BEGIN_PAUSE", pauseBegin)))
      features.append(new IndicatorFeature(("END_PAUSE", pauseEnd)))
      
      //Duration of previous word over/under
      if (begin > 0) {
        val word = words(begin-1)
        val avgDuration = if (averageDurations.contains(word)) averageDurations(word) else -1 
        val duration = prosodyFeats(begin-1)(durationIndex)
        if ((duration-avgDuration)/avgDuration > DURATION_THRESHOLD) {
          features.append(IndicatorFeature("BEGIN_DURATION_OVER"))
          if (pauseBegin == "1") {
            features.append(IndicatorFeature(("BEGIN_DURATION_OVER", "PAUSE_BEGIN")))
          }
        } else if ((avgDuration-duration)/avgDuration > DURATION_THRESHOLD) {
          features.append(IndicatorFeature("BEGIN_DURATION_UNDER"))
        }
      }

      //Duration of current word
      val word = words(begin)
      val avgDuration = if (averageDurations.contains(word)) averageDurations(word) else -1
      val duration = prosodyFeats(begin)(durationIndex)
      if ((duration-avgDuration)/avgDuration > DURATION_THRESHOLD) {
        features.append(IndicatorFeature("CURRENT_DURATION_OVER"))
        if (pauseBegin == "1") {
          features.append(IndicatorFeature(("CURRENT_DURATION_OVER", "PAUSE_BEGIN")))
        }
        if (pauseEnd == "1") {
          features.append(IndicatorFeature(("CURRENT_DURATION_OVER", "PAUSE_END")))
        }
      } else if ((avgDuration-duration)/avgDuration > DURATION_THRESHOLD) {
        features.append(IndicatorFeature("CURRENT_DURATION_UNDER"))
      }
          
      if (end < words.length) {
        val word = words(end)
        val avgDuration = if (averageDurations.contains(word)) averageDurations(word) else -1
        val duration = prosodyFeats(end)(durationIndex)
        if ((duration-avgDuration)/avgDuration > DURATION_THRESHOLD) {
          features.append(IndicatorFeature("END_DURATION_OVER"))
          if (pauseBegin == "1") {
            features.append(IndicatorFeature(("END_DURATION_OVER", "PAUSE_END")))
          }
        } else if ((avgDuration-duration)/avgDuration > DURATION_THRESHOLD) {
          features.append(IndicatorFeature("END_DURATION_UNDER"))
        }
      }
            
      allFeatures.append(features.toArray)
    }

    
    def featuresForWord(pos: Int):Array[Feature] = {
      allFeatures(pos)
      
    }
    
    def words: IndexedSeq[String] = w
  }
}
object ProsodyWordFeaturizer {

  type ProsodyLookupTable = HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]];


  def makeLookupTable(pathToTaggedSentences: String): ProsodyLookupTable = {
    val in = breeze.io.FileStreams.input(new File(pathToTaggedSentences))
    val br = new BufferedReader(new InputStreamReader(in, "UTF-8"));
    val lookupTable = new HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]]
    var thisSent = new ArrayBuffer[String];
    var thisSentFeats = new ArrayBuffer[Array[Double]];
    var count = 0
    while (br.ready()) {
      val line = br.readLine();
      if (line.trim.isEmpty) {
        if (lookupTable.contains(thisSent)) {
          count += 1
        }
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
    println("There were %s duplicate sentences".format(count));
    lookupTable;
  }

  def apply(pathsToTaggedSentences: Seq[String]) = {
    val lookupTable = pathsToTaggedSentences.map(makeLookupTable(_)).reduce(_ ++ _);
    new ProsodyWordFeaturizer(lookupTable);
  }

}
