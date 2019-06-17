setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "Geisinger H&E", "Stain 1" : "Hematoxylin", "Values 1" : "0.51346 0.77067 0.37739 ", "Stain 2" : "Eosin", "Values 2" : "0.21093 0.93879 0.27238 ", "Background" : " 221 214 220 "}');
createSelectAllObject(true);
runPlugin('qupath.imagej.detect.nuclei.WatershedCellDetection', '{"detectionImageBrightfield": "Optical density sum",  "backgroundRadius": 25.0,  "medianRadius": 3.0,  "sigma": 10.0,  "minArea": 90.0,  "maxArea": 1600.0,  "threshold": 0.2,  "maxBackground": 0.0,  "watershedPostProcess": true,  "cellExpansion": 7.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');
// Export the centroids for all detections, along with their classifications

// Set this to true to use a nucleus ROI, if available
boolean useNucleusROI = true

// Start building a String with a header
sb = new StringBuilder("Class\ty\tx\n")

// Loop through detections
int n = 0
for (detection in getDetectionObjects()) {
    // Request cell measurements & calculate integrated density
    //double cellMean = measurement(detection, "Cell: DAB OD mean")
    //double cellArea = measurement(detection, "Cell: Area")
    //double cellIntegratedDensity = cellMean * cellArea
    //println(cellIntegratedDensity)
    // Only add measurement if it's not 'Not a Number' - this implies both mean & area were available
    // Add new measurement to the measurement list of the detection
    //detection.getMeasurementList().addMeasurement("Cell: DAB integrated density", cellIntegratedDensity)
    // It's important for efficiency reasons to close the list
    //detection.getMeasurementList().closeList()
    
    def roi = detection.getROI()
    // Use a Groovy metaClass trick to check if we can get a nucleus ROI... if we need to
    // (could also use Java's instanceof qupath.lib.objects.PathCellObject)
    if (useNucleusROI && detection.metaClass.respondsTo(detection, "getNucleusROI") && detection.getNucleusROI() != null)
        roi = detection.getNucleusROI()
    // ROI shouldn't be null...
    if (roi == null)
        continue
    // Get class
    def pathClass = detection.getPathClass()
    def className = pathClass == null ? "" : pathClass.getName()
    // Get centroid
    double cx = roi.getCentroidX()
    double cy = roi.getCentroidY()
    detection.getMeasurementList().addMeasurement("Centroid X", cx)
    detection.getMeasurementList().closeList()
    detection.getMeasurementList().addMeasurement("Centroid Y", cy)
    detection.getMeasurementList().closeList()
    // Append to String
    sb.append(String.format("%s\t%.2f\t%.2f\n", className, cx, cy))
    // Count
    n++
}
fireHierarchyUpdate()
runClassifier('/data/geis_qupath/part_one/classifiers/jaya_geiss_jan_30.qpclassifier');
selectObjectsByClass(qupath.lib.objects.PathRootObject);
runPlugin('qupath.opencv.features.DelaunayClusteringPlugin', '{"distanceThreshold": 40.0,  "limitByClass": true,  "addClusterMeasurements": true}');
saveDetectionMeasurements('/data/recurrence_seq_lstm/qupath_output/', );