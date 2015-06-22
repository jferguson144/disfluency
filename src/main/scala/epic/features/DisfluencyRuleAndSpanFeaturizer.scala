package epic.features

import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import epic.parser.RuleTopology
import epic.trees.AnnotatedLabel
import epic.parser.features.IndicatorFeature
import breeze.linalg.Counter2
import scala.collection.mutable.HashMap
import breeze.linalg.Counter
import breeze.linalg.sum
import scala.collection.mutable.HashSet

class DisfluencyRuleAndSpanFeaturizer (posTags:HashMap[IndexedSeq[String],IndexedSeq[String]],
									   commonWords:HashSet[String],
									   topology: RuleTopology[AnnotatedLabel],
									   prosodyFeaturizer:SurfaceFeaturizer[String]=null) extends RuleAndSpansFeaturizer[String] {
  val rawDisfluencySpanFeaturizer = if (prosodyFeaturizer != null) {
    new DisfluencySpanFeaturizer(posTags, commonWords) + prosodyFeaturizer
  } else {
    new DisfluencySpanFeaturizer(posTags, commonWords)
  }
  val disfluencySpanFeaturizer = new PreprocessedFeaturizer(rawDisfluencySpanFeaturizer, posTags)
  val emptyArray = Array[Feature]();
  val biasArray = Array[Feature](IndicatorFeature(1));
  
  override def anchor(w: IndexedSeq[String]) : Anchoring = new Anchoring {
    val loc = disfluencySpanFeaturizer.anchor(w)
    override def words = w
    override def featuresForBinaryRule(begin: Int, split: Int, end: Int, rule: Int, ref: Int):Array[Feature] = {
      val res:ArrayBuffer[Feature] = ArrayBuffer[Feature]()
      val lc = topology.labelIndex.get(topology.leftChild(rule)).baseLabel;
      val rc = topology.labelIndex.get(topology.rightChild(rule)).baseLabel;
      val parent = topology.labelIndex.get(topology.parent(rule)).baseLabel;
      if (lc == "EDITED") {
        for (f <- loc.featuresForSpan(begin, split)) {
          res.append(IndicatorFeature((parent, f)))
        }
      } else if (rc == "EDITED") {
        for (f <- loc.featuresForSpan(split, end)) {
          res.append(IndicatorFeature((parent, f)))
        }
      } else if (parent == "EDITED") {
        for (f <- loc.featuresForSpan(begin, end)) {
          res.append(IndicatorFeature((lc, rc, f)))
        }
      }
      res.toArray
    }
    
    def featuresForUnaryRule(begin: Int, end: Int, rule: Int, ref: Int):Array[Feature] = emptyArray;
    def featuresForSpan(begin: Int, end: Int, tag: Int, ref: Int):Array[Feature] = emptyArray;
    
  }
}