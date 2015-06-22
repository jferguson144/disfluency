package epic.sequences

import epic.framework.EvaluationResult
import com.typesafe.scalalogging.slf4j.LazyLogging
import scala.collection.mutable.ArrayBuffer
import epic.trees.Span


/**
 * Object for evaluating [[epic.sequences.Segmentation]]s. Returned metrics
 * are precision, recall, and f1
 *
 * @author dlwh
 */
object SegmentationEval extends LazyLogging {
  def eval[L ,W](crf: SemiCRF[L, W], examples: IndexedSeq[Segmentation[L, W]], outsideLabel: L):Stats = {
    val ret = examples.par.aggregate(new Stats(0,0,0)) ({ (stats, gold )=>
      val guess = crf.bestSequence(gold.words, gold.id +"-guess")
      try {
        if(guess.label != gold.label)
          logger.trace(s"gold = $gold guess = $guess " +
            s"guess logPartition = ${crf.goldMarginal(guess.segments, guess.words).logPartition} " +
            s"gold logPartition =${crf.goldMarginal(gold.segments, gold.words).logPartition}")
      } catch {
        case ex: Exception => logger.debug("Can't recover gold for " + gold)
      }
      val (myStats, mergedGuess, mergedGold) = evaluateExample(Set(outsideLabel), guess, gold)
      logger.info("\nGuess: " + mergedGuess.render(badLabel=outsideLabel) + "\n Gold: " + mergedGold.render(badLabel=outsideLabel)+ "\n" + myStats)
      stats + myStats
    }, {_ + _})
    println("Guess: %s, Actual: %s, Correct: %s".format(ret.nGuess, ret.nGold, ret.nRight))
    ret
  }
  
  def evaluateExample[W, L](outsideLabel: Set[L], guess: Segmentation[L, W], gold: Segmentation[L, W]): (SegmentationEval.Stats, Segmentation[String, W], Segmentation[String,W]) = {
    val guessSet = guess.segments.filter(a => !outsideLabel(a._1)).toSet
    val guessWords = ArrayBuffer[Integer]()
    for (s <- guessSet) {
      for (i <- s._2.begin until s._2.end) {
        guessWords.append(i)
      }
    }
    val guessWordsSet = guessWords.toSet
    val mergedGuess = ArrayBuffer[Tuple2[String, Span]]()
    var prevLabel = if (guessWordsSet.contains(0)) "E" else "O"
    var start = 0
    for (i <- 1 until guess.length) {
      val curLabel = if (guessWordsSet.contains(i)) "E" else "O"
      if (curLabel != prevLabel) {
        mergedGuess.append((prevLabel, Span(start, i)))
        start = i
      }
      prevLabel = curLabel
    }
    mergedGuess.append((prevLabel, Span(start, guess.length)))
    val mergedGuessSeg = Segmentation(mergedGuess, guess.words)
    
    val goldSet = gold.segments.filter(a => !outsideLabel(a._1)).toSet
    val goldWords = ArrayBuffer[Integer]()
    for (s <- goldSet) {
      for (i <- s._2.begin until s._2.end) {
        goldWords.append(i)
      }
    }
    val goldWordsSet = goldWords.toSet
    
    val mergedGold = ArrayBuffer[Tuple2[String, Span]]()
    prevLabel = if (goldWordsSet.contains(0)) "E" else "O"
    start = 0
    for (i <- 1 until gold.length) {
      val curLabel = if (goldWordsSet.contains(i)) "E" else "O"
      if (curLabel != prevLabel) {
        mergedGold.append((prevLabel, Span(start, i)))
        start = i
      }
      prevLabel = curLabel
    }
    mergedGold.append((prevLabel, Span(start, gold.length)))
    val mergedGoldSeg = Segmentation(mergedGold, gold.words)
    
    val nRight = (guessWordsSet & goldWordsSet).size
    val myStats: Stats = new Stats(nRight, guessWordsSet.size, goldWordsSet.size) 
    (myStats, mergedGuessSeg, mergedGoldSeg)
  }

  class Stats(val nRight: Int = 0, val nGuess: Int = 0, val nGold: Int = 0) extends EvaluationResult[Stats] {
    def precision = nRight * 1.0 / nGuess
    def recall = nRight * 1.0 / nGold
    def f1 = 2 * precision * recall / (precision + recall)

    def +(stats: Stats) = {
      new Stats(nRight + stats.nRight, nGuess + stats.nGuess, nGold + stats.nGold)
    }

    override def toString = "Evaluation Result: P=%.4f R=%.4f F=%.4f".format(precision,recall,f1)

  }

}
