package epic.features

import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import epic.parser.features.IndicatorFeature

class PreprocessedWordFeaturizer(f:WordFeaturizer[String], posTagMap:HashMap[IndexedSeq[String], IndexedSeq[String]]) extends WordFeaturizer[String] with Serializable {

  override def anchor(rawW: IndexedSeq[String]): WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
	val (offsets, w, posTags) = DisfluencySpanFeaturizer.preprocess(rawW, posTagMap(rawW.map(_.toLowerCase())))
	val loc = f.anchor(w)
	override def words: IndexedSeq[String] = w
  
	override def featuresForWord(pos: Int):Array[Feature] = {
	  val nextOffset = offsets(pos+1)
	  val offset = offsets(pos)
	  val newPos = pos-offset
	  if (nextOffset - offset == 1) {
	    //Throw feature that says it's a preprocessed word
	    Array(IndicatorFeature("FILLER_WORD"))
	  } else {
	    //Throw features for new index on preprocessed words
	    loc.featuresForWord(newPos)
	  }
	}
  }
}