package epic.trees
/*
 Copyright 2012 David Hall

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
import java.io.File
import breeze.config.Help
import epic.util.ArabicNormalization
import scala.collection.mutable.ArrayBuffer

/**
 * Represents a treebank with attendant spans, binarization, etc. Used in all the parser trainers.
 *
 * @author dlwh
 */

@Help(text="Parameters for reading and processing a treebank.")
case class ProcessedTreebank(@Help(text="Location of the treebank directory")
                             path: File,
                             @Help(text="Max length for training sentences")
                             maxLength: Int = 10000,
                             @Help(text="Should we add the dev set for training, do this only for final test.")
                             includeDevInTrain: Boolean = false,
                             @Help(text="What kind of binarization to do. Options: left, right, head. Head is best.")
                             binarization: String = "head",
                             treebankType: String = "penn",
                             numSentences: Int = Int.MaxValue,
                             keepUnaryChainsFromTrain: Boolean = true,
                             debuckwalterize: Boolean = false,
                             supervisedHeadFinderPtbPath: String = "",
                             supervisedHeadFinderConllPath: String = "",
                             removePartials: Boolean = false) {

  lazy val treebank = treebankType.toLowerCase() match {
    case "penn" => Treebank.fromPennTreebankDir(path)
    case "chinese" => Treebank.fromChineseTreebankDir(path)
    case "negra" => Treebank.fromGermanTreebank(path)
    case "simple" => new SimpleTreebank(new File(path, "train.txt"), new File(path, "dev.txt"), new File(path, "test.txt"))
    case "single" => new SimpleTreebank(path, path, path)
    case "eval" => new SimpleTreebank(new File(path, "gold"), new File(path, "guess"), new File(path, "guess"))
    case "conllonto" => Treebank.fromOntonotesDirectory(path)
    case "spmrl" =>
      var trainPath: File = new File(path, "train")
      if(!trainPath.exists)
        trainPath = new File(path, "train5k")
      val train = trainPath.listFiles().filter(_.getName.endsWith("ptb"))
      val dev = new File(path, "dev").listFiles().filter(_.getName.endsWith("ptb"))
      val test = new File(path, "test").listFiles().filter(_.getName.endsWith("ptb"))
      new SimpleTreebank(train, dev, test)
    case "spmrl5k" =>
      val train = new File(path, "train5k").listFiles().filter(_.getName.endsWith("ptb"))
      val dev = new File(path, "dev").listFiles().filter(_.getName.endsWith("ptb"))
      val test = new File(path, "test").listFiles().filter(_.getName.endsWith("ptb"))
      new SimpleTreebank(train, dev, test)
    case _ => throw new RuntimeException("Unknown Treebank type")
  }

  lazy val trainTrees: IndexedSeq[TreeInstance[AnnotatedLabel, String]] = {
    var train = transformTrees(treebank.train, maxLength, collapseUnaries = true, removePartials = removePartials)
    if(includeDevInTrain) train ++= transformTrees(treebank.dev, maxLength, collapseUnaries = true, removePartials = removePartials)
    train.take(numSentences)
  }
  lazy val devTrees = transformTrees(treebank.dev, 100000, removePartials=removePartials)
  lazy val testTrees = transformTrees(treebank.test, 1000000, removePartials=removePartials)


  def transformTrees(portion: treebank.Portion, maxL: Int, collapseUnaries: Boolean = false, removePartials:Boolean=false): IndexedSeq[TreeInstance[AnnotatedLabel, String]] = {
    val binarizedAndTransformed = for (
      ((tree, words), index) <- portion.trees.zipWithIndex if words.length <= maxL;
      w2 = if(debuckwalterize) words.map(ArabicNormalization.buckwalterToUnicode) else words
    ) yield {
      val name = s"${portion.name}-$index"
      makeTreeInstance(name, tree, w2, collapseUnaries, removePartials)
    }

    val res = binarizedAndTransformed.filter(x => x != null)
    res.toIndexedSeq
  }


  def makeTreeInstance(name: String, tree: Tree[String], words: IndexedSeq[String], collapseUnaries: Boolean, removePartials:Boolean=false): TreeInstance[AnnotatedLabel, String] = {
    val offsets = Array.fill(words.length+1)(0)
    val newWords = if (removePartials) {
      val tmp = ArrayBuffer[String]()
      val keepWord = Array.fill(words.length)(true)
      for ((i, l) <- tree.leaves.filter(x=> x.span.length>0).map(x=>(x.span.begin, x.label))) {
        val w = words(i)
        if (l == "XX" || w.endsWith("-")) {
          keepWord(i) = false
          for (j <- i+1 until offsets.length) {
            offsets(j) += 1
          }
        }
      }
      for (i <- 0 until words.length) {
        if (keepWord(i)) {
          tmp.append(words(i).toLowerCase())
        }
      }
      tmp
    } else {
      words
    }
    def rec(t: Tree[String]):Tree[String] = {
      val begin = t.span.begin
      val end = t.span.end
      val newSpan = Span(begin-offsets(begin), end-offsets(end))
      if (newSpan.length == 0) {
        null
      } else {
        val newChildren = t.children.map(rec(_)).filter(_!=null)
        if (t.children.length > 0 && newChildren.length == 0) {
          null
        } else {
          Tree(t.label, newChildren, newSpan)
        }
      }
    }
    val t2 = if (removePartials) rec(tree) else tree
    
    if (t2 == null) return null
    var transformed = process(t2)
    if (transformed == null)
      return null
    if (collapseUnaries) {
      transformed = UnaryChainCollapser.collapseUnaryChains(transformed, keepChains = keepUnaryChainsFromTrain)
    }            
    TreeInstance(name, transformed, newWords)
  }

  def headRules = {
    binarization match {
      case "xbar" | "right" => HeadFinder.right[String]
      case "leftXbar" | "left" => HeadFinder.left[String]
      case "head" => if (treebankType .startsWith("spmrl")) {
        SupervisedHeadFinder.trainHeadFinderFromFiles(supervisedHeadFinderPtbPath, supervisedHeadFinderConllPath);
      } else {
        HeadFinder.collins
      }
      case _ => HeadFinder.collins
    }
  }

  val process: StandardTreeProcessor = new StandardTreeProcessor(headRules)
}



