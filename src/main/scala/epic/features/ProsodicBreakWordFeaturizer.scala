package epic.features

import epic.framework.Feature
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature
import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import scala.collection.mutable.ArrayBuffer

class ProsodicBreakWordFeaturizer private (breakLookupTable: ProsodicBreakFeaturizer.BreakLookupTable) extends WordFeaturizer[String] with Serializable {
  final val BEGIN_SENTENCE_BREAK = "4"
  final val UNKNOWN_BREAK = "X"
  def anchor(w: IndexedSeq[String]): WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
    val prosodyFeats = if (!breakLookupTable.contains(w.mkString(" "))) {
      val sliced = w.slice(0, w.length-1).mkString(" ")
      if (breakLookupTable.contains(sliced)) {
        breakLookupTable(sliced) ++ "1"
      } else {
    	  println("Sentence wasn't found in lookup table: " + w.mkString(" "))
    	  (0 until w.size).map(i => UNKNOWN_BREAK); //Some non-value for each position we're considering breaks for (split, prior to beginning, after end) 
      }
    } else {
      breakLookupTable(w.mkString(" "))
    }
    val feats = (0 until w.size).map(i => prosodyFeats(i))
    
    def featuresForWord(pos: Int): Array[Feature] = {
      val ret = ArrayBuffer[Feature]()
      val beforeBreakVal = if (pos == 0) { "START" } else { prosodyFeats(pos - 1) }
      val afterBreakVal = if (pos == words.length - 1) { "END" } else { prosodyFeats(pos) }
      if (beforeBreakVal != UNKNOWN_BREAK) {
        ret.append(IndicatorFeature(("BEFORE_BREAK", beforeBreakVal)))
      }
    	
      if (afterBreakVal != UNKNOWN_BREAK) {
        ret.append(IndicatorFeature(("AFTER_BREAK", afterBreakVal)))  
      }
      ret.toArray
    }
    
    def words: IndexedSeq[String] = w
  }
}

object ProsodicBreakWordFeaturizer {

  def apply(pathsToTaggedSentences: Seq[String]) = {
    val lookupTable = pathsToTaggedSentences.map(ProsodicBreakFeaturizer.makeLookupTable(_)).reduce(_ ++ _);
    new ProsodicBreakWordFeaturizer(lookupTable);
  }

}
