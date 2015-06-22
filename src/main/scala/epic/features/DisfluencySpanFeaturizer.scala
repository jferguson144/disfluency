package epic.features

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.Stack
import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import epic.parser.features.IndicatorFeature
import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.FileReader
import breeze.linalg.Counter2
import breeze.linalg.Counter
import breeze.linalg.sum
import breeze.collection.mutable.TriangularArray
import breeze.util.Index


class DisfluencySpanFeaturizer (posTagMap:HashMap[IndexedSeq[String], IndexedSeq[String]]/*wordTagCounts: Counter2[String, String, Double]*/, commonWords:HashSet[String]=null) extends SurfaceFeaturizer[String] with Serializable {
  
  var PREPROCESS = true
  val newPosTagMap = if (PREPROCESS) {
    val tmp = HashMap[IndexedSeq[String], IndexedSeq[String]]()
    for ((k,v) <- posTagMap) {
      val (_, newK, newV) = DisfluencySpanFeaturizer.preprocess(k,v) 
      tmp.put(newK, newV)
    }
    tmp
  } else {
    posTagMap
  }
  val wordFeaturizer = new DisfluencyWordFeaturizer(/*wordTagCounts*/newPosTagMap)
  
  var anchI = 0
  
  override def anchor(rawW: IndexedSeq[String]): SurfaceFeatureAnchoring[String] = new SurfaceFeatureAnchoring[String] {
	val WINDOW_SIZE = 3
	val MAX_SEARCH_LENGTH = 10
	val w = rawW.map(_.toLowerCase())
	val posTags = newPosTagMap(w)
    val loc = wordFeaturizer.anchor(w.map(x=>x.toLowerCase()))

    val paddedWords = Array("BEGIN_SENTENCE_TOKEN") ++ w.map(_.toLowerCase()) ++ Array("END_SENTENCE_TOKEN")
    val paddedPosTags = Array("BEGIN_POS_TOKEN") ++ /*w.map(tag(_))*/posTags ++ Array("END_POS_TOKEN")
    
    val posDuplicates = ArrayBuffer[Set[Int]]()
    val wordDuplicates = ArrayBuffer[Set[Int]]()
    for (i <- 0 until paddedWords.length) {
      val curPos = paddedPosTags(i)
      val curPosDup = ArrayBuffer[Int]()
      val curWord = paddedWords(i)
      val curWordDup = ArrayBuffer[Int]()
      for (j <- i+1 to i+WINDOW_SIZE) {
        if (j < paddedWords.length) {
          if (curPos == paddedPosTags(i)) {
            curPosDup.append(j-i)
          }
          if (curWord == paddedWords(i)) {
            curWordDup.append(j-i)
          }
        }
        posDuplicates.append(curPosDup.toSet)
        wordDuplicates.append(curWordDup.toSet)
      }
    }

    val featureCache = TriangularArray.tabulate(w.length+1) {(begin, end) => 

      val paddedBegin = begin+1
      val paddedEnd = end+1
      if (begin < end) {
      val features = ArrayBuffer[Feature]()
      val tokenFeatures = ArrayBuffer[Feature]()
      if (end-begin > 1) {
        for (f <- loc.featuresForWord(begin)) {
            tokenFeatures.append(IndicatorFeature(("B",f)))
        }
        for (i <- begin+1 until end-1) {
          for (f <- loc.featuresForWord(i)) {
            tokenFeatures.append(IndicatorFeature(("I",f)))
          }
        }
        for (f <- loc.featuresForWord(end-1)) {
          tokenFeatures.append(IndicatorFeature(("E",f)))
        }
      } else {
        for (f <- loc.featuresForWord(begin)) {
            tokenFeatures.append(IndicatorFeature(("S",f)))
        }
      }
      val length = end-begin
//      features.append(IndicatorFeature(length)) //Span length
//      features.append(IndicatorFeature(("ENDING_BIGRAM", paddedWords(paddedEnd-1), paddedWords(paddedEnd))))
//      features.append(IndicatorFeature(("SURROUNDING_BIGRAM", paddedWords(paddedBegin-1), paddedWords(paddedEnd))))
////      if (paddedWords(paddedBegin) == paddedWords(paddedEnd)) {
////        features.append(IndicatorFeature("DUPLICATE_BEGINNING"))
//        val dist = paddedEnd-paddedBegin
////        numDup = 1
      var numUncommonDup = 0
      var numCommonDup = 0
      var numPrevUncommonDup = 0
      var numPrevCommonDup = 0
      for (i <- paddedBegin until paddedEnd) {
        var flag = false
        var prevFlag = false
        for (dist <- 1 until 10) {
          if (!flag && i+dist < paddedWords.length && paddedWords(i) == paddedWords(i+dist)) {
            if (commonWords.contains(paddedWords(i))) {
              numCommonDup += 1
            } else {
              numUncommonDup+=1
            }
            flag = true
          }
          if (!prevFlag && i-dist > 0 && paddedWords(i) == paddedWords(i-dist)) {
            if (commonWords.contains(paddedWords(i))) {
              numPrevCommonDup += 1
            } else {
              numPrevUncommonDup+=1
            }
            prevFlag = true
          }
        }
      }
      var dupDist = -1
      for (dist <- 10 to 0 by -1) {
        if (paddedEnd+dist < paddedWords.length && paddedWords(paddedBegin) == paddedWords(paddedEnd+dist)) {
          dupDist = dist
        }
      }
        
      if (dupDist != -1 && dupDist+paddedEnd+length <= paddedWords.length) {
        val beginDupWindow = paddedEnd+dupDist
        
        var allDup = true
        for (i <- 1 until length) {
          val dupPos = paddedEnd+dupDist+i
          if (paddedWords(paddedBegin+i) != paddedWords(dupPos)) {
            allDup = false
          }
        }
        if (allDup) {
          features.append(IndicatorFeature(("FULL_SPAN_DUP", length)))
        }
      }
      
      var dupPosDist = -1
      for (dist <- 10 to 0 by -1) {
        if (paddedEnd+dist < paddedWords.length && paddedPosTags(paddedBegin) == paddedPosTags(paddedEnd+dist)) {
          dupDist = dist
        }
      }
        
      var numPrevDup = 0
      var numDup = 0
      
      val bucket = if (length < 10) length/2 else 5   
      features.append(IndicatorFeature(("NUM_UNCOMMON_DUPLICATES", numUncommonDup)))
      features.append(IndicatorFeature(("NUM_UNCOMMON_DUP_LENGTH", numUncommonDup,length)))      
      
      if (paddedPosTags(paddedBegin) == paddedPosTags(paddedEnd)) {
        features.append(IndicatorFeature("DUPLICATE_POS_BEGINNING"))
        var dupLength = 1
        while(paddedPosTags(paddedBegin+dupLength) == paddedPosTags(paddedEnd+dupLength)
            && dupLength < length) {
          dupLength += 1
        }
        features.append(IndicatorFeature(("BEGINNING_POS_CHAIN_LENGTH", dupLength)))
        if (dupLength > (length)/2) {
          features.append(IndicatorFeature("HALF_POS_CHAIN"))
        }
      }
      
      if (paddedWords(paddedBegin) == paddedWords(paddedEnd)) {
        features.append(IndicatorFeature("DUPLICATE_WORD_BEGINNING"))
        var dupLength = 1
        while(paddedWords(paddedBegin+dupLength) == paddedWords(paddedEnd+dupLength)
            && dupLength < length) {
          dupLength += 1
        }
        features.append(IndicatorFeature(("BEGINNING_WORD_CHAIN_LENGTH", dupLength)))
        if (dupLength > (length)/2) {
          features.append(IndicatorFeature("HALF_WORD_CHAIN"))
        }
      }      
      
      features.append(IndicatorFeature(("BEGINNING_POS_BIGRAM", paddedPosTags(paddedBegin-1), paddedPosTags(paddedBegin))))
      features.append(IndicatorFeature(("ENDING_POS_BIGRAM", paddedPosTags(paddedEnd-1), paddedPosTags(paddedEnd))))
      features.append(IndicatorFeature(("SURROUNDING_POS_BIGRAM", paddedPosTags(paddedBegin-1), paddedPosTags(paddedEnd))))
      
      (tokenFeatures ++ features).toArray
      } else {
        null
      }
    }

    override def featuresForSpan(begin: Int, end: Int): Array[Feature] = {
        featureCache(begin, end)
    }
    
    override def featuresForLabelledSpan(begin: Int, end: Int, label: String): Array[Feature] = {
        Array[Feature]()
    }
    
  }

}
object DisfluencySpanFeaturizer {
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
}