package epic.sequences

import java.io._
import breeze.config.{ Configuration, CommandLineParser }
import epic.ontonotes.{ NerType, ConllOntoReader }
import collection.mutable.ArrayBuffer
import breeze.linalg.DenseVector
import epic.framework.ModelObjective
import breeze.optimize._
import nak.data.Example
import breeze.util.Encoder
import epic.trees.Span
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.Implicits._
import epic.util.CacheBroker
import com.typesafe.scalalogging.slf4j.LazyLogging
import epic.corpora.CONLLSequenceReader
import epic.features.DisfluencyTransitionFeaturizer
import epic.features.DisfluencyWordFeaturizer
import epic.trees.TreeInstance
import scala.collection.mutable.{ Stack, ArrayBuffer, HashMap, HashSet }
import epic.trees.AnnotatedLabel
import epic.trees.Tree
import epic.trees.ProcessedTreebank
import epic.features.DisfluencySpanFeaturizer
import breeze.linalg.Counter2
import epic.framework.EvaluationResult

/*
  command line: treebank.path <directory with trees> treebank.type eval
  Directory should have guess trees in "train.txt", gold trees in "dev.txt"
  Compute token level disfluency F score given trees as input
*/
object DisfluencyFScore extends LazyLogging {
  case class Params(opt: OptParams, treebank: ProcessedTreebank, modelOut: File = new File("disfluency-model.ser.gz"), nthreads: Int = -1)
  def main(args: Array[String]) {
    val params = CommandLineParser.readIn[Params](args)
    import params._
    val (goldSegments, trainPosTags) = TrainSemiDisfluencyDetector.extractDisfluencyRepresentation(treebank.trainTrees, removePartials=true)
    val (guessedSegments, devPosTags) = TrainSemiDisfluencyDetector.extractDisfluencyRepresentation(treebank.devTrees, removePartials=true,goldTrees=treebank.trainTrees)
    
    var i=0
    val allStats = goldSegments.aggregate(new SegmentationEval.Stats(0,0,0)) ({ (stats, gold )=>
      val guess = guessedSegments(i)
      i += 1
      val (myStats,_,_) = SegmentationEval.evaluateExample(Set("O"), guess, gold)
      logger.info("\nGuess: " + guess.render(badLabel="O") + "\n Gold: " + gold.render(badLabel="O")+ "\n" + myStats)
      stats + myStats
    }, {_ + _})    

    println(allStats.toString)
    println("Predicted: " + allStats.nGuess)
    println("Actual: " + allStats.nGold)
    println("Correct: " + allStats.nRight)
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