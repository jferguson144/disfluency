package epic.features

import epic.framework.Feature
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature
import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import scala.collection.mutable.ArrayBuffer
import breeze.collection.mutable.TriangularArray

class ProsodySpanFeaturizer (prosodyLookupTable: ProsodySpanFeaturizer.ProsodyLookupTable) extends SurfaceFeaturizer[String] with Serializable {
  final val UNKNOWN_FEATURE_VAL: Double = -1.0
  final val NUM_FEATURES = 5
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
      if (duration > 0) {
        val curTotal = averageDurations.getOrElse(word, 0.0)
        val curCount = wordCounts.getOrElse(word, 0).asInstanceOf[Integer]
        averageDurations.put(word, curTotal+duration)
        wordCounts.put(word, curCount+1)
      }
    }
  }
  for ((word, total) <- averageDurations) {
    val count = wordCounts(word)
    averageDurations.put(word, total/count)
  }
  
  def anchor(w: IndexedSeq[String]): SurfaceFeatureAnchoring[String] = new SurfaceFeatureAnchoring[String] {
    val prosodyFeats: IndexedSeq[Array[Double]] = if (!prosodyLookupTable.contains(w)) {
      println("Sentence wasn't found in lookup table: " + w.mkString(" "))
      (0 until w.size).map(i => UNKNOWN_FEATURES); //Some non-value for each position we're considering breaks for (split, prior to beginning, after end)
    } else {
      prosodyLookupTable(w)
    }

    var totalNumOver = 0
    var totalNumUnder = 0
    for (i <- 0 until words.length) {
      val word = words(i)
      val avgDuration = averageDurations.getOrElse[Double](word, -1)
      val duration = prosodyFeats(i)(durationIndex)
      if ((duration-avgDuration)/avgDuration > DURATION_THRESHOLD) {
        totalNumOver += 1
      } else if ((avgDuration-duration)/avgDuration > DURATION_THRESHOLD) {
        totalNumUnder += 1
      }
    }
    
    val spanFeatures = TriangularArray.tabulate(w.length+1) {(begin, end) => 
      if (begin < end) {
   
      val featsBeforeBegin = if (begin == 0) { UNKNOWN_FEATURES } else { prosodyFeats(begin - 1) }
      val featsAfterBegin = prosodyFeats(begin)
      val featsBeforeEnd = prosodyFeats(end - 1)
      val featsAfterEnd = if (end == prosodyFeats.size) {UNKNOWN_FEATURES} else {prosodyFeats(end)}
      
      var beginPause = 0
      var endPause = 0
      var beginEnergy = 0
      var endEnergy = 0
      
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


      def binPause(pauseDuration: Double): String = {
        if (pauseDuration > 0) {
          "1"
        } else {
          "0"
        } 
      }

      val pauseBegin = if (featsBeforeBegin(pauseIndex) == UNKNOWN_FEATURE_VAL) { "x" } else { binPause(featsBeforeBegin(pauseIndex)) }
      val pauseEnd = if (end == prosodyFeats.length - 1) {
        "e"
      } else if (featsBeforeEnd(pauseIndex) == UNKNOWN_FEATURE_VAL) {
        "x"
      } else {
        binPause(featsBeforeEnd(pauseIndex))
      }

      if (pauseBegin == "1") beginPause = 1
      if (pauseEnd == "1") endPause = 1

      val features = new ArrayBuffer[Feature]
      features.append(new IndicatorFeature(("BEGIN_PAUSE", pauseBegin)))
      features.append(new IndicatorFeature(("END_PAUSE", pauseEnd)))
      
      if (begin > 0) {
        val word = words(begin-1)
        val avgDuration = averageDurations.getOrElse[Double](word,-1)
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

      if (end < words.length) {
        val word = words(end)
        val avgDuration = averageDurations.getOrElse[Double](word, -1)
        val duration = prosodyFeats(end)(durationIndex)
        if ((duration-avgDuration)/avgDuration > DURATION_THRESHOLD) {
          features.append(IndicatorFeature("END_DURATION_OVER"))
          if (pauseEnd == "1") {
            features.append(IndicatorFeature(("END_DURATION_OVER", "PAUSE_END")))
          }
        } else if ((avgDuration-duration)/avgDuration > DURATION_THRESHOLD) {
          features.append(IndicatorFeature("END_DURATION_UNDER"))
        }
      }
            
      var numPauses = 0
      var numOver = 0
      var numUnder = 0
      for (i <- begin until end) {
        if (prosodyFeats(i)(pauseIndex) > 0) {
          numPauses += 1
        }
        val word = words(i)
        val avgDuration = averageDurations.getOrElse[Double](word, -1)
        val duration = prosodyFeats(i)(durationIndex) 
        if ((duration-avgDuration)/avgDuration > DURATION_THRESHOLD) {
          numOver += 1
        } else if ((avgDuration-duration)/avgDuration > DURATION_THRESHOLD) {
          numUnder += 1
        }
        
      }
      features.append(new IndicatorFeature(("NUM_PAUSES", numPauses)))
      features.append(new IndicatorFeature(("NUM_PAUSES_LENGTH", numPauses, end-begin)))
      features.append(new IndicatorFeature(("NUM_OVER", numOver, words.length-totalNumOver)))
      features.append(new IndicatorFeature(("NUM_OVER", numOver, words.length-totalNumOver, end-begin)))
      features.append(new IndicatorFeature(("NUM_UNDER", numUnder, words.length-totalNumOver)))
      features.append(new IndicatorFeature(("NUM_UNDER", numUnder, words.length-totalNumOver, end-begin)))
      
      
      features.toArray
      } else {
        null
      }
    }

    
    def featuresForSpan(begin: Int, end:Int):Array[Feature] = {
      spanFeatures(begin, end)
      
    }
    
    def words: IndexedSeq[String] = w
  }
}
object ProsodySpanFeaturizer {

  type ProsodyLookupTable = HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]];

  def makeLookupTable(pathToTaggedSentences: String): ProsodyLookupTable = {
    val in = breeze.io.FileStreams.input(new File(pathToTaggedSentences))
    val br = new BufferedReader(new InputStreamReader(in, "UTF-8"));
    val lookupTable = new HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]]
    var thisSent = new ArrayBuffer[String];
    var thisSentFeats = new ArrayBuffer[Array[Double]];
    var count = 0
    while (br.ready()) {
      val line = br.readLine().toLowerCase();
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
    new ProsodySpanFeaturizer(lookupTable);
  }

}
