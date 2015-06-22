package epic
package features

import breeze.collection.mutable.TriangularArray
import breeze.util.Index
import epic.framework.Feature
import scala.collection.mutable
import epic.constraints.SpanConstraints
import epic.util.CacheBroker
import scala.collection.mutable.ArrayBuffer

/**
 *
 * @author dlwh
 */
trait IndexedSurfaceFeaturizer[W] {
  def featureIndex: Index[Feature]
  def featurizer: SurfaceFeaturizer[W]
  def anchor(datum: IndexedSeq[W]):IndexedSurfaceAnchoring[W]
}

trait IndexedSurfaceAnchoring[W] {
  def words: IndexedSeq[W]
  def featuresForSpan(begin: Int, end: Int):Array[Int]
  def featuresForLabelledSpan(begin: Int, end: Int, label: String):Array[Int] = Array[Int]()
}

object IndexedSurfaceFeaturizer {
  def fromData[L,W](feat: SurfaceFeaturizer[W],
                  data: IndexedSeq[IndexedSeq[W]],
                  constraintFactory: SpanConstraints.Factory[W],
                  labelIndex:Index[L]) : IndexedSurfaceFeaturizer[W]  = {
    val index = Index[Feature]()
    for(words <- data) {
      val cons = constraintFactory.get(words)
      val anch = feat.anchor(words)
      for(i <- 0 until words.length) {
        for(j <- (i+1) to (i + cons.maxSpanLengthStartingAt(i)) if cons(i, j)) {
          for (l <- 0 until labelIndex.size) {
            anch.featuresForLabelledSpan(i,j,labelIndex.get(l).toString) foreach {index.index _}
          }
          anch.featuresForSpan(i, j) foreach {index.index}
        }          
      }
    }


    val f = new MySurfaceFeaturizer[L, W](feat, constraintFactory, index, labelIndex)
    new CachedFeaturizer(f, CacheBroker().make(f.toString))
  }

  @SerialVersionUID(1L)
  class CachedFeaturizer[W](val base: IndexedSurfaceFeaturizer[W], cache: collection.mutable.Map[IndexedSeq[W], IndexedSurfaceAnchoring[W]]) extends IndexedSurfaceFeaturizer[W] with Serializable {
    def featurizer: SurfaceFeaturizer[W] = base.featurizer

    def featureIndex: Index[Feature] = base.featureIndex

    def anchor(datum: IndexedSeq[W]): IndexedSurfaceAnchoring[W] = cache.getOrElseUpdate(datum, base.anchor(datum))
  }

  @SerialVersionUID(3L)
  private class MySurfaceFeaturizer[L, W](val featurizer: SurfaceFeaturizer[W],
                                       constraintsFactory: SpanConstraints.Factory[W],
                                       val featureIndex: Index[Feature],
                                       labelIndex:Index[L]) extends IndexedSurfaceFeaturizer[W] with Serializable {
    def anchor(words: IndexedSeq[W]):IndexedSurfaceAnchoring[W]  = {
      val cons = constraintsFactory.constraints(words)
      val anch = featurizer.anchor(words)
      val spanFeatures = ArrayBuffer[TriangularArray[Array[Int]]]()
//      val spanFeatures = TriangularArray.tabulate(words.length+1){ (i, j) =>
      for (l <- 0 until labelIndex.size+1) {
        spanFeatures.append(TriangularArray.tabulate(words.length+1){ (i, j) =>
          if(cons(i,j) && i < j) {
            if (l >0) {
              stripEncode(featureIndex, anch.featuresForLabelledSpan(i, j, labelIndex.get(l-1).toString))
            }else{
              stripEncode(featureIndex, anch.featuresForSpan(i, j))
            }
          } else {
            null
          }
        })
      }

      new TabulatedIndexedSurfaceAnchoring[L, W](words, spanFeatures, labelIndex)

    }
  }

  def stripEncode(ind: Index[Feature], features: Array[Feature]) = {
    val result = mutable.ArrayBuilder.make[Int]()
    result.sizeHint(features)
    var i = 0
    while(i < features.length) {
      val fi = ind(features(i))
      if(fi >= 0)
        result += fi
      i += 1
    }
    result.result()
  }
}

@SerialVersionUID(2L)
class TabulatedIndexedSurfaceAnchoring[L, W](val words: IndexedSeq[W],
                                          spanFeatures: IndexedSeq[TriangularArray[Array[Int]]],
                                          labelIndex:Index[L]) extends IndexedSurfaceAnchoring[W] with Serializable {
  override def featuresForLabelledSpan(begin: Int, end: Int, label: String):Array[Int] = {
    spanFeatures(labelIndex(label.asInstanceOf[L])+1)(begin, end) ++ spanFeatures(0)(begin,end)
  }
  
  def featuresForSpan(begin: Int, end: Int):Array[Int] = {
    spanFeatures(0)(begin, end)
  }
}