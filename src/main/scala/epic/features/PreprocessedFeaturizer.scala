package epic.features

import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature



class PreprocessedFeaturizer(f:SurfaceFeaturizer[String], posTagMap:HashMap[IndexedSeq[String], IndexedSeq[String]]) extends SurfaceFeaturizer[String] {
val PREPROCESS = true
  
  def preprocess(rawWords:IndexedSeq[String], posTags: IndexedSeq[String]):(IndexedSeq[Int], IndexedSeq[String], IndexedSeq[String]) = {
    val words = rawWords.map(_.toLowerCase())  
    val offsets = Array.fill(words.length+1) {0}
    val newWords = ArrayBuffer[String]()
    val newPosTags = ArrayBuffer[String]() 
    var currentOffset = 0
    for (i <- 0 until words.length) {
      val posTag = posTags(i)
      var toAdd = words(i)
      if (posTag == "XX" || toAdd.endsWith("-")) {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = ""
      }
      if (words(i) == "um" || words(i) == "uh") {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = ""
      }
      if (i>0 && words(i)=="know" && words(i-1) == "you") {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = ""
      }
      if (i>0 && words(i)=="mean" && words(i-1) == "i") {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = ""
      }
      if (i<words.length-1 && words(i) == "i" && words(i+1) == "mean") {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = "" 
      }
      if (i<words.length-1 && words(i) == "you" && words(i+1) == "know") {
        for (j <- i+1 to words.length) {
          offsets(j)+=1
        }
        toAdd = ""
      }
      
      
      if (toAdd != "") {
        newWords.append(toAdd)
        newPosTags.append(posTags(i))
      }
    }
    (offsets, newWords, newPosTags)
  }
  
  override def anchor(rawW: IndexedSeq[String]): SurfaceFeatureAnchoring[String] = new SurfaceFeatureAnchoring[String] {
	val (offsets, w, posTags) = if (PREPROCESS) {
	  preprocess(rawW, posTagMap(rawW.map(_.toLowerCase())))
	} else {
	  (null, rawW, posTagMap(rawW.map(_.toLowerCase())))
	}
	val loc = f.anchor(w)
	

    
    override def featuresForLabelledSpan(begin: Int, end: Int, label: String): Array[Feature] = {
      val newBegin = if (PREPROCESS) begin-offsets(begin) else begin
      val newEnd = if (PREPROCESS) end-offsets(end) else end
      val firstWord = rawW(begin).toLowerCase()
      val prevWord = if (begin > 0) rawW(begin-1).toLowerCase() else ""
      val lastWord = rawW(end-1).toLowerCase()
      val nextWord = if (end < rawW.length) rawW(end).toLowerCase() else ""
      if (newEnd > newBegin && 
          lastWord != "uh" && 
          lastWord != "um" && 
          firstWord != "um" && 
          firstWord != "uh" &&
          !(prevWord == "i" && firstWord == "mean") && 
          !(prevWord == "you" && firstWord == "know") && 
          !(lastWord == "i" && nextWord == "mean") && 
          !(lastWord == "you" && nextWord == "know")) {
        loc.featuresForLabelledSpan(newBegin, newEnd, label)
      } else {
        Array[Feature]()
      }
    }
    override def featuresForSpan(begin: Int, end: Int): Array[Feature] = {
	  if (PREPROCESS) {
	  	val window = 3
        val newBegin = begin-offsets(begin)
        val newEnd = end-offsets(end)
        val firstWord = rawW(begin).toLowerCase()
        val prevWord = if (begin > 0) rawW(begin-1).toLowerCase() else ""
        val lastWord = rawW(end-1).toLowerCase()
        val nextWord = if (end < rawW.length) rawW(end).toLowerCase() else ""
      
        val fillerFeatures = ArrayBuffer[Feature]()
        val baseFeatures = if (newBegin == newEnd) Array[Feature]() else loc.featuresForSpan(newBegin, newEnd)
        
        if (lastWord == "uh" || lastWord == "um") {
          fillerFeatures.append(IndicatorFeature("ENDING_FILLER"))
        } else if (checkBigram(rawW, end-1, "i", "mean") ||
    	          checkBigram(rawW, end-1, "you", "know")) {
          fillerFeatures.append(IndicatorFeature("SPLIT_FILLER"))
        }
        if (firstWord == "uh" || firstWord == "um") {
          fillerFeatures.append(IndicatorFeature("STARTING_FILLER"))
        } else if (checkBigram(rawW, begin-1, "i", "mean") ||
    	          checkBigram(rawW, begin-1, "you", "know")) {
          fillerFeatures.append(IndicatorFeature("STARTING_SPLIT_FILLER"))
        }
        
        baseFeatures ++ fillerFeatures.toArray[Feature]
      } else {
        loc.featuresForSpan(begin, end)
      }
    }
  }
  def checkBigram(words:IndexedSeq[String], startingIndex:Int, firstWord:String, secondWord:String):Boolean = {
    if (startingIndex < 0) {
      false
    } else if (words.length <= startingIndex+1) {
      false
    } else {
      (words(startingIndex) == firstWord && words(startingIndex+1) == secondWord)
    }
  }
}