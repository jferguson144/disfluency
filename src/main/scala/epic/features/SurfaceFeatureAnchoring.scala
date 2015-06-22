package epic.features

import epic.framework.Feature

/**
  *
  * @author dlwh
  */
trait SurfaceFeatureAnchoring[W]  {
  def featuresForSpan(begin: Int, end: Int):Array[Feature]
  def featuresForLabelledSpan(begin: Int, end: Int, label: String):Array[Feature] = Array[Feature]()
}
