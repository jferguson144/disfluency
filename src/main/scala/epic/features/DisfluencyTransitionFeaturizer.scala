package epic.features

import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import epic.parser.features.IndicatorFeature

/*
  Just adds a bias feature for each transition
*/
class DisfluencyTransitionFeaturizer[W] extends WordFeaturizer[W] with Serializable {
  override def anchor(w: IndexedSeq[W]): WordFeatureAnchoring[W] = new WordFeatureAnchoring[W] {
    override def words: IndexedSeq[W] = w

    val feats: Array[Array[Feature]] = {
      val ret = ArrayBuffer[Array[Feature]]()
      for (word <- words) {
        val features = ArrayBuffer[Feature]()
        features.append(IndicatorFeature(1))
        ret.append(features.toArray)
      }
      ret.toArray
    }

    override def featuresForWord(pos: Int): Array[Feature] = {
      Array[Feature]()
    }
  }
}