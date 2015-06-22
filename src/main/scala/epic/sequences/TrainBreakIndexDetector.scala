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
import epic.features.BreakIndexFeaturizer
import epic.features.DisfluencyWordFeaturizer
import epic.features.ProsodySpanFeaturizer
import epic.features.ProsodyWordFeaturizer
import epic.features.ConstituentBoundaryFeaturizer

object TrainBreakIndexDetector extends LazyLogging {
  case class Params(opt: OptParams, 
		  			pathToData: String, 
		  			modelOut: File = new File("disfluency-model.ser.gz"),
		  			usePosTaggerFeatures:Boolean = false,
		  			useProsody:Boolean = false,
		  			pathToProsody:String = "",
		  			useDisfluency:Boolean = false,
		  			pathToSyntax:String = "",
		  			evalAll:Boolean = false,
		  			evalPath:String = "",
		  		    otherPath:String="",
		  		    useSyntax:Boolean = false)

  def main(args: Array[String]) {
    val params = CommandLineParser.readIn[Params](args)
    logger.info("Command line arguments for recovery:\n" + Configuration.fromObject(params).toCommandLineString)
    import params._
        
    var combinedWordFeaturizer:WordFeaturizer[String] = new BreakIndexFeaturizer()
    val otherTreebank = ProcessedTreebank(path=new File(otherPath), treebankType="simple")
    val data = extractBreakIndices(pathToData)
    val posTagMap = HashMap[IndexedSeq[String], IndexedSeq[String]]()
    val treebank = ProcessedTreebank(path=new File(pathToSyntax), treebankType="simple")
    if (useDisfluency) {
      for (t <- treebank.trainTrees) {
        posTagMap.put(t.words.map(_.toLowerCase()), t.asTaggedSequence.tags.map(_.label))
      }
      for (t <- treebank.devTrees) {
        posTagMap.put(t.words.map(_.toLowerCase()), t.asTaggedSequence.tags.map(_.label))
      }
      for (t <- treebank.testTrees) {
        posTagMap.put(t.words.map(_.toLowerCase()), t.asTaggedSequence.tags.map(_.label))
      }
      
      for (t <- otherTreebank.trainTrees) {
        posTagMap.put(t.words.map(_.toLowerCase()), t.asTaggedSequence.tags.map(_.label))
      }
      for (t <- otherTreebank.devTrees) {
        posTagMap.put(t.words.map(_.toLowerCase()), t.asTaggedSequence.tags.map(_.label))
      }
      for (t <- otherTreebank.testTrees) {
        posTagMap.put(t.words.map(_.toLowerCase()), t.asTaggedSequence.tags.map(_.label))
      }
      
    }
    
    val filteredProsodyTable = HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]]()
    val allTrees = treebank.trainTrees ++ treebank.devTrees ++ treebank.testTrees ++ otherTreebank.trainTrees ++ otherTreebank.devTrees ++ otherTreebank.testTrees
    if (useProsody) {
    
      val prosodyTable = ProsodySpanFeaturizer.makeLookupTable(pathToProsody)
      for (t <- allTrees) {
        val pFeatures = prosodyTable.getOrElse(t.words.map(_.toLowerCase()), t.words.map(x=>Array(-1.0, -1.0, -1.0)))
        val pFeaturesString = pFeatures.map(_.mkString(","))
        
        val (k,rawV) = (t.words.map(_.toLowerCase()),pFeaturesString)
        val v = rawV.map(_.split(",").map(_.toDouble))
        filteredProsodyTable.put(k,v)
      }
    }
    
    var allStats = new TaggedSequenceEval.Stats[String]()
    
    for (i <- 0 until 10) {
      val (train, dev) = splitData(data, i)
      logger.info(""+i)
      if (usePosTaggerFeatures) {
        val counts: Counter2[String, String, Double] = Counter2.count(train.flatMap(p => p.label zip p.words)).mapValues(_.toDouble)
        combinedWordFeaturizer += WordFeaturizer.goodPOSTagFeaturizer(counts)
      } 
      
      if (useDisfluency) {
        combinedWordFeaturizer += new DisfluencyWordFeaturizer(posTagMap)
      }
      
      if (useProsody) {
        combinedWordFeaturizer += new ProsodyWordFeaturizer(filteredProsodyTable)
      }

      if (useSyntax) {
        combinedWordFeaturizer += new ConstituentBoundaryFeaturizer(allTrees)
      }
      
      val transitionFeaturizer = new DisfluencyTransitionFeaturizer[String]()
      val crf = CRF.buildSimple[String](train, "TOP", wordFeaturizer = combinedWordFeaturizer, transitionFeaturizer = transitionFeaturizer, opt = opt)
      val stats = TaggedSequenceEval.eval(crf, dev)
      
      allStats += stats
      
      if (i == 9 && evalAll) {
        var k = 0
        val outText = allTrees.par.aggregate("") ({ (text, gold )=> 
          val guess = crf.bestSequence(gold.words.map(_.toLowerCase()), gold.id +"-guess")
          val out = guess.render
    	    val list = out.split(" ")
    	    val words = ArrayBuffer[String]()
    	    val tags = ArrayBuffer[String]()
    	    for (token <- list) {
    	      val fields = token.split("/")
    	      words.append(fields(0))
    	      tags.append(fields(1))
    	    }
          text + "%s\n%s\n\n".format(words.mkString(" ").toLowerCase(), tags.mkString(" "))
        }, {_ + _})
        val writer = new PrintWriter("allBreaks", "UTF-8")
        writer.println(outText)
        writer.close()
      
      }
      
    }
    println("Final Stats: " + allStats)
    println("Confusion Matrix:\n" + allStats.confusion)

  }

  def splitData(data: IndexedSeq[TaggedSequence[String, String]], index:Int): (IndexedSeq[TaggedSequence[String, String]], IndexedSeq[TaggedSequence[String, String]]) = {
    val train = ArrayBuffer[TaggedSequence[String, String]]()
    val dev = ArrayBuffer[TaggedSequence[String, String]]()
    val beginDev = data.length/10*index
    val endDev = if (index == 9) data.length else data.length/10*(index+1)
    for (i <- 0 until data.length) {
      if (i >= beginDev && i < endDev) {
        dev.append(data(i))
      } else {
        train.append(data(i))
      }
    }
    (train, dev)
  }
  
  def extractBreakIndices(pathToData: String): IndexedSeq[TaggedSequence[String, String]] = {
    val data = ArrayBuffer[TaggedSequence[String, String]]()
    val in = breeze.io.FileStreams.input(new File(pathToData))
    val br = new BufferedReader(new InputStreamReader(in, "UTF-8"));
    
    while (br.ready()) {
      val sentence = br.readLine().trim().toLowerCase()
      val breaks = br.readLine().trim()
      br.readLine()
      val processedBreaks:IndexedSeq[String] = breaks.split(" ").map(x=>{
       val prefix = x.substring(0,1)
       val suffix = if (x.length() > 1)  x.substring(1,2) else ""
       if (suffix == "-") {
         prefix
       }  else {
         prefix+suffix
       }
      }.toString)
      val datum = new TaggedSequence(processedBreaks, sentence.split(" "))
      data.append(datum)
    }
    
    data
  }
}
