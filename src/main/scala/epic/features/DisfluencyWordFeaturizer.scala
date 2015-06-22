package epic.features

import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import epic.parser.features.IndicatorFeature
import scala.Tuple2
import scala.collection.mutable.HashMap
import breeze.linalg.Counter2
import breeze.linalg.Counter
import breeze.linalg.sum
import scala.collection.mutable.HashSet

class DisfluencyWordFeaturizer(posTagMap:HashMap[IndexedSeq[String], IndexedSeq[String]]/*wordTagCounts: Counter2[String, String, Double]*/) extends WordFeaturizer[String] with Serializable {
  
  override def anchor(w: IndexedSeq[String]): WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
    val MAX_SEARCH_DISTANCE = 15
    val WINDOW_SIZE = 1
    val FUZZY_MATCH_THRESHOLD = 0.9
    override def words: IndexedSeq[String] = w

    def fuzzyMatch(w1:String, w2:String):Boolean = {
      val numMatchingChars = (w1.toSet&w2.toSet).size
      val score = 2*numMatchingChars/(w1.length()+w2.length()).toFloat
      score > FUZZY_MATCH_THRESHOLD
    }
    
    val feats: Array[Array[Feature]] = {
      val ret = ArrayBuffer[Array[Feature]]()
      var currentInd = 0
      val posTags = posTagMap(w)//w.map(tag(_))
      val words = w.map(word => word.toLowerCase)
      val duplicateBigramPos = ArrayBuffer[Tuple2[Int, Int]]() 
      val duplicates = Array.fill(words.length)(-1)
      val fuzzyDuplicates = Array.fill(words.length)(-1)
      val posDuplicates = Array.fill(words.length)(-1)
      for (word <- words) {
        val features = ArrayBuffer[Feature]()
        val currentPosTag = posTags(currentInd)
        var forwardDuplicate = -1
        var forwardPosDuplicate = -1
        var partialWordDistance = -1
        
        for (i <- MAX_SEARCH_DISTANCE to 1 by -1) {
          if (currentInd + i < words.length) {
            if (words(currentInd + i) == word) {
              features.append(IndicatorFeature(("FORWARD_DUPLICATE", i)))
              features.append(IndicatorFeature(("FORWARD_DUPLICATE", i, word)))
              forwardDuplicate = i
            }
            if (posTags(currentInd + i) == currentPosTag) {
              features.append(IndicatorFeature(("FORWARD_POS_DUPLICATE", i)))
              features.append(IndicatorFeature(("FORWARD_POS_DUPLICATE", i, currentPosTag)))
              forwardPosDuplicate = i
            }

          }
        }
        duplicates(currentInd) = forwardDuplicate
        posDuplicates(currentInd) = forwardPosDuplicate
        for (i <- MAX_SEARCH_DISTANCE to 1 by -1) {
          if (currentInd - i > 0) {
            if (words(currentInd - i) == word) {
              features.append(IndicatorFeature(("BACKWARD_DUPLICATE", i)))
              features.append(IndicatorFeature(("BACKWARD_DUPLICATE", i, word)))
            }
            if (posTags(currentInd - i) == currentPosTag) {
              features.append(IndicatorFeature(("BACKWARD_POS_DUPLICATE", i)))
              features.append(IndicatorFeature(("BACKWARD_POS_DUPLICATE", i, currentPosTag)))
            }
          }
        }
        if (currentInd > 0) {
          var prevWordDuplicate = duplicates(currentInd-1)
          var prevWordPosDuplicate = posDuplicates(currentInd-1)
          val prevInd = currentInd - 1
          val prevWord = words(prevInd)
          val prevPosTag = posTags(prevInd)
          
          if (prevWordDuplicate == forwardDuplicate && prevWordDuplicate != -1) {
            features.append(IndicatorFeature("DUPLICATE_CHAIN"))
            features.append(IndicatorFeature(("DUPLICATE_CHAIN", posTags(prevInd))))
            features.append(IndicatorFeature(("DUPLICATE_CHAIN", posTags(prevInd), posTags(currentInd))))
            duplicateBigramPos.append((prevInd, prevWordDuplicate))
          } else if (prevWordDuplicate == forwardDuplicate-1 && prevWordDuplicate != -1) {
            features.append(IndicatorFeature("DUPLICATE_CHAIN"))
            features.append(IndicatorFeature(("DUPLICATE_CHAIN", posTags(prevInd))))
            features.append(IndicatorFeature(("DUPLICATE_CHAIN", posTags(prevInd), posTags(currentInd))))
            duplicateBigramPos.append((prevInd, prevWordDuplicate))
          } else if (currentInd > 1 && duplicates(currentInd-2)+1 == forwardDuplicate) {
            features.append(IndicatorFeature("DUPLICATE_CHAIN"))
            features.append(IndicatorFeature(("DUPLICATE_CHAIN", posTags(prevInd-1))))
            features.append(IndicatorFeature(("DUPLICATE_CHAIN", posTags(prevInd-1), posTags(currentInd))))
            duplicateBigramPos.append((currentInd-2, duplicates(currentInd-2)))
          }  
        } else {
          features.append(IndicatorFeature("BEGIN_SENTENCE"))
        }

        if (currentInd >= words.length - 1) {
          features.append(IndicatorFeature("END_SENTENCE"))
        }

        val startInd = math.max(currentInd-WINDOW_SIZE, 0)
        val endInd = math.min(currentInd+WINDOW_SIZE, words.length-1)
        for (i <- startInd to endInd) {
          val unigram = words(i)
          val posUnigram = posTags(i)
          features.append(IndicatorFeature(("UNIGRAM", i-currentInd, unigram)))
          features.append(IndicatorFeature(("POS_UNIGRAM", i-currentInd, posUnigram)))
          if (i<endInd) {
            //add bigram features
        	val nextUnigram = words(i+1)
        	val nextPosUnigram = posTags(i+1)
        	features.append(IndicatorFeature(("BIGRAM", i-currentInd, unigram, nextUnigram)))
            features.append(IndicatorFeature(("POS_BIGRAM", i-currentInd, posUnigram, nextPosUnigram)))
          }
        }
        
        currentInd += 1
        ret.append(features.toArray)
      }
      ret.toArray
    }
    
    override def featuresForWord(pos: Int): Array[Feature] = {
      feats(pos)
    }
  }
}