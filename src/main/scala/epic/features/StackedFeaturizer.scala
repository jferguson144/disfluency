package epic.features

import scala.collection.mutable.HashMap
import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import epic.parser.features.IndicatorFeature

/*
  Takes in mapping of sentences to predicted disfluencies
  Output combination features for original span and features for span with predicted disfluencies removed
*/
class StackedFeaturizer (f:SurfaceFeaturizer[String], disfluencyMap:HashMap[IndexedSeq[String], IndexedSeq[Boolean]]) extends SurfaceFeaturizer[String] {
  
  override def anchor(rawW: IndexedSeq[String]): SurfaceFeatureAnchoring[String] = new SurfaceFeatureAnchoring[String] {
	val (offsets, w, ignored) = StackedFeaturizer.preprocess(rawW, disfluencyMap(rawW.map(_.toLowerCase())))
	val stackedLoc = f.anchor(w)
	val loc = f.anchor(rawW)
	

    
    override def featuresForLabelledSpan(begin: Int, end: Int, label: String): Array[Feature] = {
      val newBegin = begin-offsets(begin)
      val newEnd = end-offsets(end)
      val originalFeatures = loc.featuresForLabelledSpan(begin, end, label) 
      if (newEnd-newBegin == 0) {
        originalFeatures
      } else {
        originalFeatures ++ stackedLoc.featuresForLabelledSpan(newBegin, newEnd, label).map(x=>IndicatorFeature(("STACKED", x)))
      }
    }
    override def featuresForSpan(begin: Int, end: Int): Array[Feature] = {
      val newBegin = begin-offsets(begin)
      val newEnd = end-offsets(end)
      val originalFeatures = loc.featuresForSpan(begin, end)
      if (newEnd-newBegin == 0) {
        originalFeatures
      } else {
        originalFeatures ++ stackedLoc.featuresForSpan(newBegin, newEnd).map(x=>IndicatorFeature(("STACKED", x)))
      }
    }
  }
}

object StackedFeaturizer {
  def preprocess(rawWords:IndexedSeq[String], disfluencies: IndexedSeq[Boolean], posTags: IndexedSeq[String]=null):(IndexedSeq[Int], IndexedSeq[String], IndexedSeq[String]) = {
    val words = rawWords.map(_.toLowerCase())  
    val offsets = Array.fill(words.length+1) {0}
    val newWords = ArrayBuffer[String]()
    val newPosTags = ArrayBuffer[String]() 
    var currentOffset = 0
    for (i <- 0 until words.length) {
      var toAdd = words(i)
      val posTag = if (posTags == null) "" else posTags(i)
      if (disfluencies(i)) {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = ""
      }
      if (toAdd != "") {
        newWords.append(toAdd)
        newPosTags.append(posTag)
      }
    }
    (offsets, newWords, newPosTags)
  }
}