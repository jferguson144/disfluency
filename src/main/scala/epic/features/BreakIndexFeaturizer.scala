package epic.features

import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer

class BreakIndexFeaturizer extends WordFeaturizer[String] with Serializable {
  
  override def anchor(w: IndexedSeq[String]): WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
    
    override def words: IndexedSeq[String] = w
    
    val feats: Array[Array[Feature]] = {
      val ret = ArrayBuffer[Array[Feature]]()
      for (currentInd <- 0 until words.length) {
        val word = words(currentInd)
        val features = ArrayBuffer[Feature]()
        
        ret.append(features.toArray)
      }
      ret.toArray
    }
    
    override def featuresForWord(pos: Int): Array[Feature] = {
      feats(pos)
    }
  }
}