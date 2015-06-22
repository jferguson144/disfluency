package epic.features

import epic.trees.TreeInstance
import epic.trees.AnnotatedLabel
import epic.framework.Feature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.Stack
import epic.trees.Tree
import epic.parser.features.IndicatorFeature

class ConstituentBoundaryFeaturizer(trees:IndexedSeq[TreeInstance[AnnotatedLabel, String]]) extends WordFeaturizer[String] with Serializable {
  //Need a mapping from word->constituent type/level if at boundary
  
  val featureMap = HashMap[IndexedSeq[String], Array[IndexedSeq[Feature]]]()
  
  for (tree <- trees) {
    val words = tree.asTaggedSequence.words.map(x=>x.toLowerCase())
    val currentLeftSide = Array.fill(words.length)(ArrayBuffer[Feature]())
    val features = Array.fill(words.length)(ArrayBuffer[Feature]())
    val fringe = Stack[Tree[AnnotatedLabel]]()
    fringe.push(tree.label)
    val closed = HashSet[Tuple2[Int, Int]]()
    while (!fringe.isEmpty) {
      val current = fringe.pop()
      if (!closed.contains((current.span.begin, current.span.end))) {
        closed.add((current.span.begin, current.span.end))
        val start = current.span.begin
        if (tree.tree.leftHeight == 1) {
          if (start > 0) {
            features(start-1).append(IndicatorFeature(("LEFT_BEFORE", tree.tree.leftHeight, current.label.baseLabel)))
          }
          features(start).append(IndicatorFeature(("LEFT_AFTER", tree.tree.leftHeight, current.label.baseLabel)))
        }
        val end = current.span.end
        if (tree.tree.leftHeight == 1) {
          if (end < words.length) {
            features(end).append(IndicatorFeature(("RIGHT_AFTER", tree.tree.leftHeight, current.label.baseLabel)))
          }
          features(end-1).append(IndicatorFeature(("RIGHT_BEFORE", tree.tree.leftHeight, current.label.baseLabel)))
        }
        
        for (child <- current.children) {
          fringe.push(child)
        }
      }
    }
    featureMap.put(words, features.toArray)
  }
  
  
  override def anchor(w: IndexedSeq[String]): WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
    override def words: IndexedSeq[String] = w
    val feats: Array[Array[Feature]] = featureMap(words).map(_.toArray)
    override def featuresForWord(pos: Int): Array[Feature] = {
      feats(pos)
    }
  }
}