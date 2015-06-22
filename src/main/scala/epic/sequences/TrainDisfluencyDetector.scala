package epic.sequences

import breeze.optimize.FirstOrderMinimizer.OptParams
import java.io._
import epic.trees.{ AnnotatedLabel, ProcessedTreebank, TreeInstance, Tree, Span }
import breeze.config.{ Configuration, CommandLineParser }
import breeze.util.Encoder
import epic.util.CacheBroker
import com.typesafe.scalalogging.slf4j.LazyLogging
import scala.collection.mutable.{ Stack, ArrayBuffer, HashSet }
import epic.features.{ DisfluencyTransitionFeaturizer, DisfluencyWordFeaturizer }
import scala.collection.mutable.HashMap
import breeze.linalg.Counter2
import epic.features.WordFeaturizer
import epic.features.ProsodicBreakWordFeaturizer
import epic.features.TagSpanShapeGenerator
import epic.features.PreprocessedWordFeaturizer
import epic.features.DisfluencySpanFeaturizer

object TrainDisfluencyDetector extends LazyLogging {
  case class Params(opt: OptParams, 
		  			treebank: ProcessedTreebank, 
		  			modelOut: File = new File("disfluency-model.ser.gz"),
		  			usePosTaggerFeatures:Boolean = false,
		  			useBreaks:Boolean = false,
		  			useProsody:Boolean = false,
		  			pathsToProsody:String = "")

  def main(args: Array[String]) {
    val params = CommandLineParser.readIn[Params](args)
    logger.info("Command line arguments for recovery:\n" + Configuration.fromObject(params).toCommandLineString)
    import params._
    val (train, trainPosTags) = extractDisfluencyWordRepresentation(treebank.trainTrees)
    val (dev, devPosTags) = extractDisfluencyWordRepresentation(treebank.devTrees)
    
    val tagCounts = TagSpanShapeGenerator.makeStandardLexicon(treebank.trainTrees)
    for ((k,v) <- devPosTags.iterator) {
      trainPosTags.put(k,v)
    }
    
    val newPosTagMap = HashMap[IndexedSeq[String], IndexedSeq[String]]()
    for ((k,v) <- trainPosTags) {
      val (_, newK, newV) = DisfluencySpanFeaturizer.preprocess(k,v) 
      newPosTagMap.put(newK, newV)
    }
    
    val internalFeaturizer = new DisfluencyWordFeaturizer(newPosTagMap)
    val wordFeaturizer = new PreprocessedWordFeaturizer(internalFeaturizer, trainPosTags)
    lazy val breakWordFeaturizer = ProsodicBreakWordFeaturizer(pathsToProsody.split(","))
    lazy val prosodyWordFeaturizer = ProsodicBreakWordFeaturizer(pathsToProsody.split(","))
    val counts: Counter2[String, String, Double] = Counter2.count(train.flatMap(p => p.label zip p.words)).mapValues(_.toDouble)
    
    var combinedWordFeaturizer:WordFeaturizer[String] = wordFeaturizer
    
    if (useProsody) {
      combinedWordFeaturizer += prosodyWordFeaturizer
    }
    
    if (useBreaks) {
      combinedWordFeaturizer += breakWordFeaturizer
    }
    
    if (usePosTaggerFeatures) {
      combinedWordFeaturizer += WordFeaturizer.goodPOSTagFeaturizer(counts)
    } 
    val transitionFeaturizer = new DisfluencyTransitionFeaturizer[String]()
    val crf = CRF.buildSimple(train, AnnotatedLabel("TOP"), wordFeaturizer = combinedWordFeaturizer, transitionFeaturizer = transitionFeaturizer, opt = opt)
    
    val stats = TaggedSequenceEval.eval(crf, dev)
    
    println("Final Stats: " + stats)
    println("Confusion Matrix:\n" + stats.confusion)


  }

  def extractDisfluencyWordRepresentation(trees: IndexedSeq[TreeInstance[AnnotatedLabel, String]]): (IndexedSeq[TaggedSequence[String, String]], HashMap[IndexedSeq[String], IndexedSeq[String]]) = {
    val (segmentations, posTags) = TrainSemiDisfluencyDetector.extractDisfluencyRepresentation(trees, removePartials=true, goldTrees=null)
    val ret = ArrayBuffer[TaggedSequence[String, String]]()
    val useExtendedTagSet = true
    for (s <- segmentations) {
      val words = s.words
      val tags = ArrayBuffer[String]()
      for ((label, span) <- s.segments) {
        if (span.length > 0) {
          if (useExtendedTagSet) {
            if (span.length > 1) {
              if (label == "E") {
                tags.append("B")
              } else {
                tags.append("OB")
              }
               for (i <- span.begin+1 until span.end-1) {
                if (label == "E") {
                  tags.append("I")
                } else {
                  tags.append("OI")
                }
              }
              if (label == "E") {
                tags.append("E")
              } else {
                tags.append("OE")
              }
            } else {
              if (label == "E") {
                tags.append("S")
              } else {
                tags.append("OS")
              }
            }
          } else {
            if (label == "E") {
              tags.append("B")
            } else {
              tags.append("OB")
            }
            for (i <- span.begin+1 until span.end-1) {
              if (label == "E") {
                tags.append("I")
              } else {
                tags.append("OI")
              }
            }
          }
        }
      }
      ret.append(TaggedSequence(tags, words))
    }
    (ret, posTags)
  }
}



