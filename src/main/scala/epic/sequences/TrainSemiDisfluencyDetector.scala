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
import epic.preprocess.TreebankTokenizer
import epic.corpora.CONLLSequenceReader
import epic.features.DisfluencyTransitionFeaturizer
import epic.features.DisfluencyWordFeaturizer
import epic.trees.TreeInstance
import scala.collection.mutable.{ Stack, ArrayBuffer, HashMap }
import epic.trees.AnnotatedLabel
import epic.trees.Tree
import epic.trees.ProcessedTreebank
import epic.features.DisfluencySpanFeaturizer
import breeze.linalg.Counter2
import epic.features.WordFeatureAnchoring
import epic.framework.Feature
import epic.features.WordFeaturizer
import epic.parser.features.IndicatorFeature
import scala.collection.mutable.HashSet
import epic.constraints.LabeledSpanConstraints
import epic.features.ProsodySpanFeaturizer
import epic.features.TagSpanShapeGenerator
import epic.trees.SimpleTreebank
import breeze.util.Index
import epic.features.PreprocessedFeaturizer
import epic.features.ProsodicBreakFeaturizer
import epic.features.SurfaceFeaturizer
import epic.features.StackedFeaturizer

object TrainSemiDisfluencyDetector extends LazyLogging {

case class Params(opt: OptParams,
				  treebank: ProcessedTreebank, 
				  modelOut: File = new File("disfluency-model.ser.gz"), 
				  nthreads: Int = -1,
				  pathsToDisfluency: String = "",
				  useBreaks: Boolean = false,
				  pathToBreaks: String = "",
				  useRawProsody: Boolean = false,				  
				  pathToProsody: String = "",
				  useStacked: Boolean = false,				  
				  pathToDisfl: String = "",
				  doCrossValidation: Boolean = false,
				  maxEditSpanLength:Int = -1,
                                  posTagPath:String = "",
                                  uncommonThreshold:Int = 50)
  def main(args: Array[String]) {
    val params = CommandLineParser.readIn[Params](args)
    logger.info("Command line arguments for recovery:\n" + Configuration.fromObject(params).toCommandLineString)
    import params._

    val (train, allPosTags) = extractDisfluencyRepresentation(treebank.trainTrees, maxEditSpanLength=maxEditSpanLength, removePartials=true)    
    val (dev, goldDevPosTags) = extractDisfluencyRepresentation(treebank.devTrees, maxEditSpanLength=maxEditSpanLength, removePartials=true)

    if (posTagPath != "") {
      val devPosTags = makePosTagMap(posTagPath, treebank.devTrees)
      for ((k,v) <- devPosTags.iterator) {
        allPosTags.put(k,v)
      }
    } else {
      for ((k,v) <- goldDevPosTags.iterator) {
        allPosTags.put(k,v)
      }
    }    
    
    //Provide empty featurizer to prevent using default
    val wordFeaturizer = new WordFeaturizer[String] {
      override def anchor(w:IndexedSeq[String]) : WordFeatureAnchoring[String] = new WordFeatureAnchoring[String] {
        override def words = w
        override def featuresForWord(pos: Int): Array[Feature] = {
          Array[Feature]()
        }
      }
    }

    val commonWords = extractUnigramCounts(train, uncommonThreshold)
    
    val filteredProsodyTable = HashMap[IndexedSeq[String], IndexedSeq[Array[Double]]]()
    if (useRawProsody) {
      val prosodyTable = ProsodySpanFeaturizer.makeLookupTable(pathToProsody)
      for (t <- treebank.trainTrees ++ treebank.devTrees) {
        val pFeatures = prosodyTable.getOrElse(t.words.map(_.toLowerCase()), t.words.map(x=>Array(-1.0, -1.0, -1.0)))
        val pFeaturesString = pFeatures.map(_.mkString(","))
        
        val (k,rawV) = preprocess(t.words.map(x=>x.toLowerCase),t.asTaggedSequence.tags.map(_.label), removePartials=true, preprocessFiller=true, pFeaturesString)
        val v = rawV.map(_.split(",").map(_.toDouble))
        filteredProsodyTable.put(k,v)
      }
    } 
    lazy val prosodySpanFeaturizer = new ProsodySpanFeaturizer(filteredProsodyTable) 
    
    val filteredBreakTable = HashMap[String, IndexedSeq[String]]()
    if (useBreaks) {
    
      val breakTable = ProsodicBreakFeaturizer.makeLookupTable(pathToBreaks)
      for (t <- treebank.trainTrees ++ treebank.devTrees) {
        val pFeatures = breakTable.getOrElse(t.words.mkString(" "), t.words.map(x=>"X"))
        
        val (k,v) = preprocess(t.words.map(x=>x.toLowerCase),t.asTaggedSequence.tags.map(_.label), removePartials=true, preprocessFiller=true, pFeatures)
        filteredBreakTable.put(k.mkString(" "),v)
      }
    }
    lazy val prosodicBreakFeaturizer = new ProsodicBreakFeaturizer(filteredBreakTable)

    val filteredDisflTable = HashMap[IndexedSeq[String], IndexedSeq[Boolean]]()
    if (useStacked) {      
      val disflTable = ProsodicBreakFeaturizer.makeLookupTable(pathToDisfl)
      for (t <- treebank.trainTrees ++ treebank.devTrees) {
        val pFeatures = disflTable.getOrElse(t.words.mkString(" ").toLowerCase(), t.words.map(x=>"X"))
        
        val (k,v) = preprocess(t.words.map(x=>x.toLowerCase),t.asTaggedSequence.tags.map(_.label), removePartials=true, preprocessFiller=true, pFeatures)
        val (words,posTags) = preprocess(t.words.map(x=>x.toLowerCase),t.asTaggedSequence.tags.map(_.label), removePartials=true, preprocessFiller=true)
        val disfluencies = v.map(x=>if (x=="1") true else false)
        filteredDisflTable.put(k,disfluencies)
        val (ignored, filteredWords, filteredPosTags) = StackedFeaturizer.preprocess(k, disfluencies, posTags)
        allPosTags.put(filteredWords, filteredPosTags)
      }
    }
    var spanFeaturizer:SurfaceFeaturizer[String] = new DisfluencySpanFeaturizer(allPosTags, commonWords=commonWords)
    
    if (useBreaks) {
      spanFeaturizer += prosodicBreakFeaturizer
    }

    if (useRawProsody) {
      spanFeaturizer += prosodySpanFeaturizer
    }
    
    if (useStacked) {
      spanFeaturizer = new StackedFeaturizer(spanFeaturizer, filteredDisflTable)
    }
     
    val combinedFeaturizer = new PreprocessedFeaturizer(spanFeaturizer, allPosTags)
    println("Creating model...")
    
    val model = new SegmentationModelFactory("B", "O", wordFeaturizer=wordFeaturizer, spanFeaturizer=combinedFeaturizer).makeModel(train)
    var trainSet = train
    var devSet = dev
    val numFolds = if (doCrossValidation) 10 else 1
    
    for (i <- 0 until numFolds) {  
      if (doCrossValidation) {
        println("Iteration %s of %s".format(i, numFolds)) 
        val foldStart = (train.length/numFolds.toDouble*i).toInt
        val foldEnd = (train.length/numFolds.toDouble*(i+1)).toInt
        trainSet = train.slice(0, foldStart) ++ train.slice(foldEnd, train.length)
        devSet = train.slice(foldStart, foldEnd)
      } else {
        model.training = true
      }
      val obj = new ModelObjective(model, trainSet, params.nthreads)
      val cached = new CachedBatchDiffFunction(obj)

      def eval(state: DenseVector[Double]) {
        val crf = model.extractCRF(state)
        println("Eval + " + SegmentationEval.eval(crf, devSet, "O"))
      }
//      val it = params.opt.iterations(cached.withRandomBatches(params.opt.batchSize), obj.initialWeightVector(randomize = false))
      val it = params.opt.iterations(cached.withScanningBatches(params.opt.batchSize), obj.initialWeightVector(randomize = false))

      val finalState = it.last.x

      if (!doCrossValidation) {
        model.cacheFeatureWeights(finalState)
      }
      model.training = false
      eval(finalState)
    }
    println("Finished")
    
//    breeze.util.writeObject(params.modelOut, model.extractCRF(finalState))

  }

  def makePosTagMap(path:String, goldTrees:IndexedSeq[TreeInstance[AnnotatedLabel, String]]):HashMap[IndexedSeq[String], IndexedSeq[String]] = {
    val excessiveTreebank = new ProcessedTreebank(new File(path), treebankType="single")
    val res = HashMap[IndexedSeq[String], IndexedSeq[String]]()
    for (i <- 0 until excessiveTreebank.trainTrees.length) {
      val tree = excessiveTreebank.trainTrees(i)
      val goldTree = goldTrees(i)
      val words = tree.asTaggedSequence.words.map(x=>x.toLowerCase())
      val posTagSeq = tree.asTaggedSequence.tags.map(_.label)
      val goldPosTagSeq = goldTree.asTaggedSequence.tags.map(_.label)
      val (k, v) = preprocess(words, goldPosTagSeq, removePartials=true, preprocessFiller=false, posTagSeq=posTagSeq)
      res.put(k,v)
    }
    res
  }

  def extractUnigramCounts(segments:IndexedSeq[Segmentation[String, String]], threshold:Integer): HashSet[String] = {
    val res=HashSet[String]()
    val tmpCounts = HashMap[String, Integer]()
    for (s <- segments) {
      for (word <- s.words) {
        if (!res.contains(word)) {
          val count = tmpCounts.getOrElse(word, 0).asInstanceOf[Int]
          val newVal = 1 + count
          tmpCounts.put(word, newVal)
          if (newVal >= threshold) {
            res.add(word)
            tmpCounts.remove(word)
          }
        }
      }
    }
    
    res
  }

  def preprocess(words:IndexedSeq[String], goldPosTagSeq:IndexedSeq[String], removePartials:Boolean, preprocessFiller:Boolean, posTagSeq:IndexedSeq[String]=null):(IndexedSeq[String],IndexedSeq[String]) = { 
    val newWords=  ArrayBuffer[String]()
    val filteredPosTagSeq = ArrayBuffer[String]()
    for (i <- 0 until words.length) {
        var toAdd = words(i)
        var toAddPos = if (posTagSeq == null) goldPosTagSeq(i) else posTagSeq(i)
        if (removePartials && (goldPosTagSeq(i)=="XX" || words(i).endsWith("-"))) {
          toAdd = ""
        } else if (preprocessFiller) {
          if (toAdd == "uh" || toAdd == "um"){ 
            toAdd = "" 
          }
          if (i>0 && toAdd=="know" && words(i-1) == "you") {
            toAdd = ""
          }
          if (i>0 && toAdd=="mean" && words(i-1) == "i") {
            toAdd = ""
          }
          if (i < words.length-1 && toAdd == "you" && words(i+1) == "know") {
            toAdd = ""
          }
          if (i < words.length-1 && toAdd == "i" && words(i+1) == "mean") {
            toAdd = ""
          }
        }
        if (toAdd != "") {
          newWords.append(toAdd)
          filteredPosTagSeq.append(toAddPos)
        }
      }
    (newWords, filteredPosTagSeq)
  }
  def extractDisfluencyRepresentation(trees: IndexedSeq[TreeInstance[AnnotatedLabel, String]], 
                                      removePartials:Boolean=false, 
                                      goldTrees: IndexedSeq[TreeInstance[AnnotatedLabel, String]]=null,
                                      maxEditSpanLength:Int= -1): (IndexedSeq[Segmentation[String, String]], HashMap[IndexedSeq[String], IndexedSeq[String]]) = {
    val ret = ArrayBuffer[Segmentation[String, String]]()
    val fringe = Stack[Tree[AnnotatedLabel]]()
    val posTags: HashMap[IndexedSeq[String], IndexedSeq[String]] = new HashMap[IndexedSeq[String], IndexedSeq[String]]

    for (k <- 0 until trees.length) {
      val tree = trees(k)
      val words = tree.asTaggedSequence.words.map(x=>x.toLowerCase())
      val posTagSeq = tree.asTaggedSequence.tags.map(_.label)
      val goldPosTagSeq = if (goldTrees != null) {
        val goldTree = goldTrees(k)
        goldTree.asTaggedSequence.tags.map(_.label)
      } else {
        posTagSeq
      }

      fringe.push(tree.label)
      val editedSpans = ArrayBuffer[Span]()
      val closed = HashSet[Tuple2[Int, Int]]()
      while (!fringe.isEmpty) {
        val current = fringe.pop()
        if (current.label.label == "EDITED" && !closed.contains((current.span.begin, current.span.end))) {
          editedSpans.append(current.span)
          closed.add((current.span.begin, current.span.end))
        }
        for (child <- current.children) {
          fringe.push(child)
        }
      }

      val offsets = Array.fill(words.length+1) {0}
      val newSegments = {
        val tmp = ArrayBuffer[Span]()
        if (removePartials) {
          for (i <- 0 until words.length) {
            if (words(i).endsWith("-") || goldPosTagSeq(i) == "XX") {
              for (j <- i+1 to words.length) {
                offsets(j)+=1
              }
            }
          }
        }
        for (s <- editedSpans) {
    	  val newS = Span(s.begin - offsets(s.begin), s.end-offsets(s.end))
    	  if (newS.length > 0 && !tmp.contains(newS)) {
    	    tmp.append(newS)
    	  }
    	}
    	tmp
      }
      val (newWords, filteredPosTagSeq) = preprocess(words, goldPosTagSeq, removePartials, false, posTagSeq)  
      
      posTags.put(newWords, filteredPosTagSeq)
      def spanComp(s1: Span, s2: Span) = s1.begin < s2.begin
      val sortedEditedSpans = newSegments.sortWith(spanComp)
      var start = 0
      val segments = ArrayBuffer[(String, Span)]()
      for (span <- sortedEditedSpans) {
        if (span.begin > start) {
          //Add outside span
          segments.append(("O", Span(start, span.begin)))
        }
        //Add edited span
        segments.append(("E", span))
        //Update start
        start = Math.max(start, span.end)
      }
      if (start != newWords.length) {
        //Add final outside span
        segments.append(("O", Span(start, newWords.length)))
      }
      if (newWords.length > 0) {
        val flattenedSegments = ArrayBuffer[(String, Span)]()
        val editedWords = ArrayBuffer[Integer]()
        for (s <- segments) {
          if (s._1 == "E") {
            for (i <- s._2.begin until s._2.end) {
              editedWords.append(i)
            }
          }  
        }
        val editedWordsSet = editedWords.toSet
        var prevLabel = if (editedWordsSet.contains(0)) "E" else "O"
        var spanStart = 0
        for (i <- 1 until newWords.length+1) {
          val curLabel = if (editedWordsSet.contains(i)) "E" else if (i==newWords.length) "" else "O"
          if (i==newWords.length && curLabel != "") throw new Error("wee")
          if (curLabel != prevLabel && spanStart != -1) {
            var spanLength = i-spanStart
            var newStart = spanStart
            var newEnd = math.min(i, newStart+maxEditSpanLength)
            while (prevLabel == "E" && maxEditSpanLength > 0 && spanLength > maxEditSpanLength) {
              spanLength -= maxEditSpanLength
              flattenedSegments.append(("E", Span(newStart, newEnd)))
              newStart = newEnd
              newEnd = math.min(i, newStart+maxEditSpanLength)
            }
            if (newStart < i)
              flattenedSegments.append((prevLabel, Span(newStart, i)))
            spanStart = i
          }
          prevLabel = curLabel
        }
        ret.append(Segmentation(flattenedSegments, newWords))
      }
    }
    (ret, posTags)
  }
  
}